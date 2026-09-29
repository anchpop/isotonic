use crate::{Point, Weight};
use num_traits::Float;

/// Ordering imposed on the fitted responses as x increases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Direction {
    Ascending,
    Descending,
}

/// Fitted observations in domain order. Fitting needs no arithmetic on x.
/// Responses must be finite, and weights finite and positive.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IsotonicRegression<X, Y = X, W = f64> {
    direction: Direction,
    points: Vec<Point<X, Y, W>>,
}
impl<X: PartialOrd + Copy, Y: Float, W: Weight> IsotonicRegression<X, Y, W> {
    pub fn new(points: &[Point<X, Y, W>], direction: Direction) -> Self {
        Self {
            points: isotonic(points, direction),
            direction,
        }
    }
    /// Input must be sorted by x. Incomparable coordinates panic.
    pub fn new_sorted(points: &[Point<X, Y, W>], direction: Direction) -> Self {
        Self {
            points: isotonic_presorted(points.to_vec(), direction),
            direction,
        }
    }
    pub fn new_ascending(points: &[Point<X, Y, W>]) -> Self {
        Self::new(points, Direction::Ascending)
    }
    pub fn new_descending(points: &[Point<X, Y, W>]) -> Self {
        Self::new(points, Direction::Descending)
    }
    pub fn new_ascending_sorted(points: &[Point<X, Y, W>]) -> Self {
        Self::new_sorted(points, Direction::Ascending)
    }
    pub fn new_descending_sorted(points: &[Point<X, Y, W>]) -> Self {
        Self::new_sorted(points, Direction::Descending)
    }
}
impl<X, Y, W> IsotonicRegression<X, Y, W> {
    pub fn points(&self) -> &[Point<X, Y, W>] {
        &self.points
    }
    pub fn into_points(self) -> Vec<Point<X, Y, W>> {
        self.points
    }
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
    pub fn direction(&self) -> Direction {
        self.direction
    }
}

pub(crate) fn isotonic<X: PartialOrd + Copy, Y: Float, W: Weight>(
    points: &[Point<X, Y, W>],
    direction: Direction,
) -> Vec<Point<X, Y, W>> {
    let mut ordered = points.to_vec();
    ordered.sort_unstable_by(|left, right| {
        left.x
            .partial_cmp(&right.x)
            .expect("incomparable x coordinates")
    });
    isotonic_presorted(ordered, direction)
}

pub(crate) fn isotonic_presorted<X: PartialOrd + Copy, Y: Float, W: Weight>(
    mut result: Vec<Point<X, Y, W>>,
    direction: Direction,
) -> Vec<Point<X, Y, W>> {
    for point in &result {
        assert!(
            point.x.partial_cmp(&point.x).is_some(),
            "incomparable x coordinate"
        );
    }
    for pair in result.windows(2) {
        assert!(
            pair[0].x <= pair[1].x,
            "x coordinates must be sorted and comparable"
        );
    }
    let n = result.len();

    // Single-pass stack-based PAV algorithm without centroid accumulation.
    // Each pool is (start_index, end_index, sum_y_weighted, sum_weight).
    // We process points left-to-right, merging adjacent pools that violate monotonicity.
    let mut pools: Vec<(usize, usize, Y, Y)> = Vec::with_capacity(n);

    for (i, point) in result.iter().enumerate().take(n) {
        let w: Y = point.weight.to_coord();
        let wy = *point.y() * w;

        pools.push((i, i + 1, wy, w));

        // Merge backwards while monotonicity is violated
        while pools.len() >= 2 {
            let len = pools.len();
            let (_, _, sum_y1, w1) = pools[len - 2];
            let (_, _, sum_y2, w2) = pools[len - 1];

            let avg1 = sum_y1 / w1;
            let avg2 = sum_y2 / w2;

            let violates = match direction {
                Direction::Ascending => avg1 > avg2,
                Direction::Descending => avg1 < avg2,
            };

            if violates {
                let (start, _, _, _) = pools[len - 2];
                let (_, end, _, _) = pools[len - 1];
                let merged = (start, end, sum_y1 + sum_y2, w1 + w2);
                pools[len - 2] = merged;
                pools.pop();
            } else {
                break;
            }
        }
    }

    // Apply pooled y-values back to all points
    for &(start, end, sum_y, sum_w) in &pools {
        let new_y = sum_y / sum_w;
        for point in result[start..end].iter_mut() {
            point.y = new_y;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnitWeight;
    #[test]
    fn test_pav_preserves_all_points() {
        // Test case from the bug description
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 10.0),
            Point::new(2.0, 9.0),
            Point::new(3.0, 8.0),
            Point::new(4.0, 20.0),
        ];

        let regression = IsotonicRegression::new_ascending(points);
        let sorted_points = regression.points();

        // All points must be preserved
        assert_eq!(
            sorted_points.len(),
            5,
            "All 5 input points should be preserved"
        );

        // X-coordinates must remain unchanged
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[1].x(), 1.0);
        assert_eq!(*sorted_points[2].x(), 2.0);
        assert_eq!(*sorted_points[3].x(), 3.0);
        assert_eq!(*sorted_points[4].x(), 4.0);

        // Y-coordinates should be pooled for violating points
        assert_eq!(*sorted_points[0].y(), 1.0); // First point unchanged
        assert_eq!(*sorted_points[1].y(), 9.0); // Pooled average of 10, 9, 8
        assert_eq!(*sorted_points[2].y(), 9.0); // Same pooled value
        assert_eq!(*sorted_points[3].y(), 9.0); // Same pooled value
        assert_eq!(*sorted_points[4].y(), 20.0); // Last point unchanged
    }

    #[test]
    fn test_weighted_pav_preserves_all_points() {
        // Test with weighted points
        let points = &[
            Point::new_with_weight(0.0, 5.0, 1.0),
            Point::new_with_weight(1.0, 3.0, 2.0),
            Point::new_with_weight(2.0, 4.0, 1.0),
            Point::new_with_weight(3.0, 10.0, 1.0),
        ];

        let regression = IsotonicRegression::new_ascending(points);
        let sorted_points = regression.points();

        // All points must be preserved
        assert_eq!(sorted_points.len(), 4);

        // X-coordinates unchanged
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[1].x(), 1.0);
        assert_eq!(*sorted_points[2].x(), 2.0);
        assert_eq!(*sorted_points[3].x(), 3.0);

        // Y-values should be pooled where monotonicity is violated
        assert!((sorted_points[0].y() - 11.0 / 3.0).abs() < 0.0001);
        assert!((sorted_points[1].y() - 11.0 / 3.0).abs() < 0.0001);
        assert_eq!(*sorted_points[2].y(), 4.0);
        assert_eq!(*sorted_points[3].y(), 10.0);
    }

    #[test]
    fn test_into_points() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(points);
        let owned_points = regression.into_points();
        assert_eq!(owned_points.len(), 3);
        assert_eq!(*owned_points[0].x(), 0.0);
        assert_eq!(*owned_points[2].x(), 2.0);
    }

    #[test]
    fn test_unit_weight_size() {
        assert_eq!(
            std::mem::size_of::<Point<f64, f64, UnitWeight>>(),
            16,
            "Point<f64, f64, UnitWeight> should be 16 bytes (two f64s)"
        );
        assert_eq!(
            std::mem::size_of::<Point<f64>>(),
            24,
            "Point<f64> (with f64 weight) should be 24 bytes"
        );
    }

    #[test]
    fn test_unit_weight_ascending() {
        let points: Vec<Point<f64, f64, UnitWeight>> = [
            (2.0, 1.0),
            (5.0, 7.0),
            (8.0, 3.0),
            (14.0, 9.0),
            (20.0, 12.0),
        ]
        .into_iter()
        .map(Point::from)
        .collect();
        let fit = IsotonicRegression::new_ascending(&points);
        assert_eq!(
            fit.points().iter().map(|p| *p.y()).collect::<Vec<_>>(),
            [1.0, 5.0, 5.0, 9.0, 12.0]
        );
        assert_eq!(fit.len(), points.len());
    }
}
