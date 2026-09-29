use crate::cast;
use crate::isotonic_regression::{Direction, IsotonicRegression, isotonic, isotonic_presorted};
use crate::point::Point;
use crate::weight::Weight;
use eytzinger::SliceExt;
use num_traits::Float;

/// A smoothed regression using box filter on an isotonic regression.
///
/// The smoothed function is piecewise quadratic and preserves monotonicity.
/// Evaluation and inversion are O(log n) operations.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SmoothRegression<X: Float, Y: Float = X> {
    /// Segment boundaries in Eytzinger order
    boundaries: Vec<X>,
    /// Quadratic coefficients (a, b, c) for each segment where y = at² + bt + c and t = x - boundary
    coeffs: Vec<(Y, Y, Y)>,
    /// Precomputed y-values at boundaries for inversion
    boundary_ys: Vec<Y>,
    /// Minimum x value in the domain
    x_min: X,
    /// Maximum x value in the domain
    x_max: X,
}

impl<X: Float, Y: Float> SmoothRegression<X, Y> {
    /// Create a smoothed regression directly from raw points.
    ///
    /// This runs isotonic regression (PAV) and then smooths in one step,
    /// avoiding any intermediate allocations.
    ///
    /// # Arguments
    /// * `points` - The raw data points
    /// * `direction` - Ascending or descending regression
    /// * `window_half_width` - Half-width of the box filter window (w)
    ///
    /// # Examples
    ///
    /// ```
    /// use isotonic::{Point, SmoothRegression};
    /// use isotonic::isotonic_regression::Direction;
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let smooth = SmoothRegression::new(&points, Direction::Ascending, 0.2);
    /// ```
    pub fn new<W: Weight>(
        points: &[Point<X, Y, W>],
        direction: Direction,
        window_half_width: X,
    ) -> SmoothRegression<X, Y> {
        let iso_points = isotonic(points, direction);
        Self::build(iso_points, window_half_width)
    }

    /// Create a smoothed regression directly from pre-sorted points.
    ///
    /// The caller must ensure the points are sorted in non-decreasing order by x.
    ///
    /// # Arguments
    /// * `points` - The raw data points, sorted by x
    /// * `direction` - Ascending or descending regression
    /// * `window_half_width` - Half-width of the box filter window (w)
    ///
    /// # Examples
    ///
    /// ```
    /// use isotonic::{Point, SmoothRegression};
    /// use isotonic::isotonic_regression::Direction;
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let smooth = SmoothRegression::new_sorted(&points, Direction::Ascending, 0.2);
    /// ```
    pub fn new_sorted<W: Weight>(
        points: &[Point<X, Y, W>],
        direction: Direction,
        window_half_width: X,
    ) -> SmoothRegression<X, Y> {
        let iso_points = isotonic_presorted(points.to_vec(), direction);
        Self::build(iso_points, window_half_width)
    }

    /// Create a smoothed regression from an isotonic regression (by value).
    ///
    /// Takes ownership of the regression to avoid cloning — the sorted points
    /// are used directly with zero copies.
    ///
    /// # Arguments
    /// * `regression` - The isotonic regression to smooth
    /// * `window_half_width` - Half-width of the box filter window (w)
    ///
    /// # Examples
    ///
    /// ```
    /// use isotonic::{Point, IsotonicRegression, SmoothRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points);
    /// let smooth = SmoothRegression::from_regression(regression, 0.2);
    /// ```
    pub fn from_regression<W: Weight>(
        regression: IsotonicRegression<X, Y, W>,
        window_half_width: X,
    ) -> SmoothRegression<X, Y> {
        let points = regression.into_points();
        Self::build(points, window_half_width)
    }

    /// Core builder: takes sorted isotonic points (any weight type) and builds
    /// the piecewise quadratic smooth regression.
    ///
    /// Uses a fused single-pass approach: boundaries are generated via 2-way merge
    /// (only x_i ± w, since x_i are not true knots of the convolution), then all
    /// boundary/midpoint y-values and coefficients are computed in one loop using
    /// 4 cursors into the cumulative integral — eliminating intermediate Vec
    /// allocations.
    fn build<W: Weight>(
        points: Vec<Point<X, Y, W>>,
        window_half_width: X,
    ) -> SmoothRegression<X, Y> {
        if points.is_empty() {
            return SmoothRegression {
                boundaries: Vec::new(),
                coeffs: Vec::new(),
                boundary_ys: Vec::new(),
                x_min: X::zero(),
                x_max: X::zero(),
            };
        }

        let x_min = *points.first().unwrap().x();
        let x_max = *points.last().unwrap().x();
        let w = window_half_width;
        let two_w = cast::<f64, Y>(2.0) * cast::<X, Y>(w);
        let two = cast::<f64, Y>(2.0);

        // Step 1: Build cumulative integral (O(n))
        let cumulative = CumulativeIntegral::new(&points);

        // Step 2: Generate sorted boundaries via 2-way merge of (x_i - w) and (x_i + w)
        // These are the only true knots of the box-filter convolution; x_i itself is
        // not a knot so including it just adds redundant segments.
        let mut boundaries = {
            let n = points.len();
            let mut result = Vec::with_capacity(2 * n);
            let (mut ia, mut ib) = (0usize, 0usize);
            // a[i] = points[i].x - w,  b[i] = points[i].x + w  (both sorted)
            let a_val = |i: usize| *points[i].x() - w;
            let b_val = |i: usize| *points[i].x() + w;

            loop {
                let va = if ia < n { Some(a_val(ia)) } else { None };
                let vb = if ib < n { Some(b_val(ib)) } else { None };

                let val = match (va, vb) {
                    (None, None) => break,
                    (Some(a), None) => {
                        ia += 1;
                        a
                    }
                    (None, Some(b)) => {
                        ib += 1;
                        b
                    }
                    (Some(a), Some(b)) => {
                        if a.partial_cmp(&b).unwrap() != std::cmp::Ordering::Greater {
                            ia += 1;
                            a
                        } else {
                            ib += 1;
                            b
                        }
                    }
                };

                // Inline domain filter + dedup
                if val < x_min {
                    continue;
                }
                if val > x_max {
                    break; // Both sequences are sorted, so everything after is also > x_max
                }
                if result.last().is_none_or(|&last| last != val) {
                    result.push(val);
                }
            }
            // Ensure x_min and x_max are always included as boundaries.
            // When the smoothing window is large relative to the domain, all
            // x_i - w values may fall below x_min (and x_i + w above x_max),
            // leaving no boundaries near the domain edges. Without these anchors,
            // queries near the edges fall back to distant segments and evaluate
            // quadratics far outside their intended range.
            if result.first().is_none_or(|&first| first != x_min) {
                result.insert(0, x_min);
            }
            if result.last().is_none_or(|&last| last != x_max) {
                result.push(x_max);
            }
            result
        };

        let num_segments = boundaries.len().saturating_sub(1);

        // Step 3: Fused single-pass — compute boundary ys, midpoint ys, and coefficients
        // using 4 independent cursors into the cumulative integral.
        // Each cursor advances forward (amortized O(1) per eval).
        let mut coeffs = Vec::with_capacity(num_segments);
        let mut boundary_ys = Vec::with_capacity(boundaries.len());

        // 4 cursors: one for each sorted evaluation sequence
        let mut cur_bm = 0usize; // boundary - w
        let mut cur_bp = 0usize; // boundary + w
        let mut cur_mm = 0usize; // midpoint - w
        let mut cur_mp = 0usize; // midpoint + w

        // Compute first boundary y
        if !boundaries.is_empty() {
            let em = cumulative.eval_at_cursor(&mut cur_bm, boundaries[0] - w);
            let ep = cumulative.eval_at_cursor(&mut cur_bp, boundaries[0] + w);
            boundary_ys.push((ep - em) / two_w);
        }

        for i in 0..num_segments {
            let b0 = boundaries[i];
            let b1 = boundaries[i + 1];
            let mid = (b0 + b1) / cast::<Y, X>(two);

            // Boundary y at b1 (b0's y is already in boundary_ys[i])
            let em = cumulative.eval_at_cursor(&mut cur_bm, b1 - w);
            let ep = cumulative.eval_at_cursor(&mut cur_bp, b1 + w);
            let y1 = (ep - em) / two_w;
            boundary_ys.push(y1);

            // Midpoint y
            let em = cumulative.eval_at_cursor(&mut cur_mm, mid - w);
            let ep = cumulative.eval_at_cursor(&mut cur_mp, mid + w);
            let ym = (ep - em) / two_w;

            // Fit quadratic in shifted coordinates: y = a*t² + b*t + c, where t = x - b0.
            // This guarantees c = y0 exactly and avoids catastrophic cancellation
            // when x values are large (especially important for f32).
            let y0 = boundary_ys[i];
            let h: Y = cast(b1 - b0);
            let dy_m = ym - y0; // delta-y at midpoint (t = h/2)
            let dy_1 = y1 - y0; // delta-y at right boundary (t = h)
            let coeff = if h.abs() < cast::<f64, Y>(1e-30) {
                (Y::zero(), Y::zero(), y0)
            } else {
                let a = cast::<f64, Y>(2.0) * (dy_1 - cast::<f64, Y>(2.0) * dy_m) / (h * h);
                let b = (cast::<f64, Y>(4.0) * dy_m - dy_1) / h;
                (a, b, y0)
            };
            coeffs.push(coeff);
        }

        // Eytzingerize boundaries and parallel arrays for cache-friendly lookup.
        //
        // In sorted order: coeffs[i] is for segment [boundaries[i], boundaries[i+1]).
        // The last boundary (index n-1) has no segment, so we pad coeffs with a dummy
        // to make it the same length as boundaries, then permute both identically.
        // After eytzingerization, interpolate finds the highest boundary <= x via
        // eytzinger search and uses the coeff at the same index.
        let n = boundaries.len();
        if n > 1 {
            // Pad coeffs to length n: the last sorted boundary has no segment.
            // Use a constant coeff that returns boundary_ys[n-1] when t=0.
            let last_y = *boundary_ys.last().unwrap();
            coeffs.push((Y::zero(), Y::zero(), last_y));
            debug_assert_eq!(coeffs.len(), n);

            // Build a (boundary, coeff, boundary_y) tuple array sorted by boundary,
            // then eytzingerize it as a unit.
            let mut combined: Vec<(X, (Y, Y, Y), Y)> = boundaries
                .iter()
                .zip(coeffs.iter())
                .zip(boundary_ys.iter())
                .map(|((&b, &c), &y)| (b, c, y))
                .collect();
            combined.eytzingerize(&mut eytzinger::permutation::InplacePermutator);

            // Unpack back
            for (i, (b, c, y)) in combined.into_iter().enumerate() {
                boundaries[i] = b;
                coeffs[i] = c;
                boundary_ys[i] = y;
            }
        }

        SmoothRegression {
            boundaries,
            coeffs,
            boundary_ys,
            x_min,
            x_max,
        }
    }

    /// Evaluate the smoothed function at a given x value.
    ///
    /// # Examples
    ///
    /// ```
    /// use isotonic::{Point, IsotonicRegression, SmoothRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points);
    /// let smooth = SmoothRegression::from_regression(regression, 0.2);
    /// let y = smooth.interpolate(1.5);
    /// ```
    pub fn interpolate(&self, x: X) -> Option<Y> {
        if self.coeffs.is_empty() {
            return None;
        }

        // Clamp x to the valid domain
        let x_clamped = x.max(self.x_min).min(self.x_max);

        // Eytzinger search to find the segment: highest boundary <= x_clamped
        let (lte, _gt) = self
            .boundaries
            .eytzinger_interpolative_search_by(|&b| b.partial_cmp(&x_clamped).unwrap());

        let i = match lte {
            Some(idx) => idx.min(self.coeffs.len() - 1),
            None => 0,
        };

        let (a, b, c) = self.coeffs[i];
        let t: Y = cast(x_clamped - self.boundaries[i]);
        Some(a * t * t + b * t + c)
    }

    /// Invert the smoothed function: given y, find x.
    ///
    /// # Examples
    ///
    /// ```
    /// use isotonic::{Point, IsotonicRegression, SmoothRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points);
    /// let smooth = SmoothRegression::from_regression(regression, 0.2);
    /// let x = smooth.invert(1.5);
    /// ```
    pub fn invert(&self, y: Y) -> Option<X> {
        if self.coeffs.is_empty() {
            return None;
        }

        // boundary_ys is in eytzinger order (same permutation as boundaries).
        // For ascending regressions, boundary_ys in sorted order is ascending.
        // For descending, it's descending. The eytzinger search expects the
        // underlying sorted order to be ascending, so we flip the comparison
        // for descending regressions.
        let ascending = if self.boundary_ys.len() >= 2 {
            // Check first and last in sorted order by looking at x_min/x_max boundaries
            let y_at_min = self.interpolate(self.x_min).unwrap_or(Y::zero());
            let y_at_max = self.interpolate(self.x_max).unwrap_or(Y::zero());
            y_at_max >= y_at_min
        } else {
            true
        };

        let (lte, _gt) = if ascending {
            self.boundary_ys
                .eytzinger_interpolative_search_by(|&by| by.partial_cmp(&y).unwrap())
        } else {
            // For descending, reverse: find highest by >= y (i.e., search with flipped comparison)
            self.boundary_ys
                .eytzinger_interpolative_search_by(|&by| y.partial_cmp(&by).unwrap())
        };

        let i = match lte {
            Some(idx) => idx.min(self.coeffs.len() - 1),
            None => 0,
        };

        let (a, b, c) = self.coeffs[i];
        let x0 = self.boundaries[i];

        // Coefficients are in shifted form: y = a*t² + b*t + c, where t = x - x0
        let epsilon = cast::<f64, Y>(1e-12);
        if a.abs() < epsilon {
            // Linear segment: y = b*t + c → t = (y - c) / b
            Some(x0 + cast::<Y, X>((y - c) / b))
        } else {
            // Quadratic: a*t² + b*t + (c - y) = 0
            let c_shifted = c - y;
            let discriminant = b * b - cast::<f64, Y>(4.0) * a * c_shifted;

            if discriminant < Y::zero() {
                return Some(X::zero());
            }

            let sqrt_disc = discriminant.sqrt();
            let two_a = cast::<f64, Y>(2.0) * a;

            // Roots in shifted coordinates, then convert back
            let t1 = (-b + sqrt_disc) / two_a;
            let t2 = (-b - sqrt_disc) / two_a;
            let root1 = x0 + cast::<Y, X>(t1);
            let root2 = x0 + cast::<Y, X>(t2);

            // Pick the root that's in bounds
            // We don't have easy access to the "next boundary in sorted order" from
            // eytzinger layout, so use domain bounds as fallback
            let x_lo = x0;
            let x_hi = self.x_max;

            if root1 >= x_lo && root1 <= x_hi {
                Some(root1)
            } else if root2 >= x_lo && root2 <= x_hi {
                Some(root2)
            } else {
                let center = (x_lo + x_hi) / cast::<f64, X>(2.0);
                let dist1 = (root1 - center).abs();
                let dist2 = (root2 - center).abs();
                if dist1 < dist2 {
                    Some(root1)
                } else {
                    Some(root2)
                }
            }
        }
    }
}

/// Helper struct for computing cumulative integrals of a piecewise linear function
struct CumulativeIntegral<X: Float, Y: Float> {
    xs: Vec<X>,
    ys: Vec<Y>,
    cumulative_areas: Vec<Y>,
}

impl<X: Float, Y: Float> CumulativeIntegral<X, Y> {
    fn new<W: Weight>(points: &[Point<X, Y, W>]) -> Self {
        let n = points.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut cumulative_areas = Vec::with_capacity(n);

        // Neumaier summation for the running cumulative area.
        // Prevents drift when many small trapezoid areas are added to a
        // large running total (especially important for f32).
        let mut sum = Y::zero();
        let mut comp = Y::zero(); // compensation term
        cumulative_areas.push(sum);

        for (i, point) in points.iter().enumerate() {
            xs.push(*point.x());
            ys.push(*point.y());

            if i > 0 {
                let dx: Y = cast(xs[i] - xs[i - 1]);
                let avg_y = (ys[i] + ys[i - 1]) / cast::<f64, Y>(2.0);
                let area = dx * avg_y;

                let t = sum + area;
                if sum.abs() >= area.abs() {
                    comp = comp + ((sum - t) + area);
                } else {
                    comp = comp + ((area - t) + sum);
                }
                sum = t;
                cumulative_areas.push(sum + comp);
            }
        }

        CumulativeIntegral {
            xs,
            ys,
            cumulative_areas,
        }
    }

    /// Evaluate the cumulative integral F(x) at a single point, advancing the
    /// cursor forward. Callers must supply queries in non-decreasing order for
    /// correct cursor advancement.
    ///
    /// The function is extended as constant beyond its domain:
    /// - For t < x_min: f(t) = y_min
    /// - For t > x_max: f(t) = y_max
    fn eval_at_cursor(&self, cursor: &mut usize, x: X) -> Y {
        let n = self.xs.len();
        let x_min = self.xs[0];
        let x_max = *self.xs.last().unwrap();

        if x <= x_min {
            return self.ys[0] * cast::<X, Y>(x - x_min);
        }
        if x >= x_max {
            return *self.cumulative_areas.last().unwrap()
                + *self.ys.last().unwrap() * cast::<X, Y>(x - x_max);
        }

        // Advance cursor so that xs[cursor] <= x < xs[cursor+1]
        while *cursor + 1 < n && self.xs[*cursor + 1] <= x {
            *cursor += 1;
        }

        if self.xs[*cursor] == x {
            self.cumulative_areas[*cursor]
        } else {
            let x0 = self.xs[*cursor];
            let x1 = self.xs[*cursor + 1];
            let y0 = self.ys[*cursor];
            let y1 = self.ys[*cursor + 1];
            let dx: Y = cast(x - x0);
            let slope = (y1 - y0) / cast::<X, Y>(x1 - x0);
            let y_at_x = y0 + slope * dx;
            self.cumulative_areas[*cursor] + dx * (y0 + y_at_x) / cast::<f64, Y>(2.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::Point;

    #[test]
    fn test_smooth_regression_basic() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points);
        let smooth = SmoothRegression::from_regression(regression, 0.1);

        // Should be close to linear in the middle
        let y = smooth.interpolate(1.0).unwrap();
        assert!((y - 1.0).abs() < 0.2, "y={} should be close to 1.0", y);
    }

    #[test]
    fn test_smooth_regression_monotonic() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 5.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points);
        let smooth = SmoothRegression::from_regression(regression, 0.1);

        // Check monotonicity in the interior (away from boundaries)
        let mut prev_y = 0.0f64;
        for i in 1..29 {
            // Skip near boundaries
            let x = i as f64 * 0.1;
            if let Some(y) = smooth.interpolate(x) {
                assert!(
                    y >= prev_y - 0.01,
                    "Not monotonic at x={:?}: y={}, prev_y={}",
                    x,
                    y,
                    prev_y
                );
                prev_y = y;
            }
        }
    }

    #[test]
    fn test_smooth_invert() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points);
        let smooth = SmoothRegression::from_regression(regression, 0.2);

        // Test round trip
        let test_x = 1.5;
        let y = smooth.interpolate(test_x).unwrap();
        let recovered_x = smooth.invert(y).unwrap();
        assert!((recovered_x - test_x).abs() < 0.1);
    }

    #[test]
    fn test_smooth_invert_descending() {
        // Test with descending regression to ensure quadratic root selection works
        let points = vec![
            Point::new(0.0, 5.0),
            Point::new(1.0, 4.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 2.0),
            Point::new(4.0, 1.0),
        ];
        let regression = IsotonicRegression::new_descending(&points);
        let smooth = SmoothRegression::from_regression(regression, 0.3);

        // Test round trip for descending
        let test_x = 2.5;
        let y = smooth.interpolate(test_x).unwrap();
        let recovered_x = smooth.invert(y).unwrap();
        assert!(
            (recovered_x - test_x).abs() < 0.2,
            "Round trip failed: x={} -> y={} -> x={}",
            test_x,
            y,
            recovered_x
        );

        // Test specific inversions
        let y_mid = 3.0;
        let x_mid = smooth.invert(y_mid).unwrap();
        let verify_y = smooth.interpolate(x_mid).unwrap();
        assert!(
            (verify_y - y_mid).abs() < 0.1,
            "Inversion verification failed: y={} -> x={} -> y={}",
            y_mid,
            x_mid,
            verify_y
        );
    }

    #[test]
    fn test_monotonicity_near_boundaries() {
        // Boss Claude's counterexample: f(x) = 10x - 5 on [0, 1]
        // This goes from -5 to +5, testing monotonicity with negative values
        let points = vec![
            Point::new(0.0, -5.0),
            Point::new(0.5, 0.0),
            Point::new(1.0, 5.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points);
        let smooth = SmoothRegression::from_regression(regression, 0.3);

        // Check monotonicity near left boundary
        let mut prev_y = smooth.interpolate(0.0).unwrap();
        for i in 1..=10 {
            let x = i as f64 * 0.1;
            let y = smooth.interpolate(x).unwrap();
            assert!(
                y >= prev_y - 1e-10,
                "Not monotonic at x={}: y={} < prev_y={}",
                x,
                y,
                prev_y
            );
            prev_y = y;
        }

        // Check monotonicity across entire domain
        prev_y = smooth.interpolate(0.0).unwrap();
        for i in 1..=100 {
            let x = i as f64 * 0.01;
            if x <= 1.0 {
                let y = smooth.interpolate(x).unwrap();
                assert!(
                    y >= prev_y - 1e-10,
                    "Not monotonic at x={}: y={} < prev_y={}",
                    x,
                    y,
                    prev_y
                );
                prev_y = y;
            }
        }
    }

    #[test]
    fn test_smooth_new_direct() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let smooth = SmoothRegression::new(&points, Direction::Ascending, 0.2);
        let y = smooth.interpolate(1.0).unwrap();
        assert!((y - 1.0).abs() < 0.2, "y={} should be close to 1.0", y);
    }

    #[test]
    fn test_smooth_new_sorted_direct() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let smooth = SmoothRegression::new_sorted(&points, Direction::Ascending, 0.2);
        let y = smooth.interpolate(1.0).unwrap();
        assert!((y - 1.0).abs() < 0.2, "y={} should be close to 1.0", y);
    }
}
