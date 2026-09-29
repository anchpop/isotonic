use isotonic::isotonic_regression::Direction;
use isotonic::weight::Weight;
use isotonic::{IsotonicRegression, Point, SmoothRegression, UnitWeight};
use num_traits::Float;

/// Evaluate a piecewise-linear function defined by sorted points at a given x.
/// Extended as constant beyond the domain.
fn eval_piecewise_linear(points: &[(f64, f64)], x: f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    if x <= points[0].0 {
        return points[0].1;
    }
    if x >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }
    let mut lo = 0;
    let mut hi = points.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if points[mid].0 <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (x0, y0) = points[lo];
    let (x1, y1) = points[hi];
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Brute-force box-filter with Neumaier summation.
/// With enough subdivisions on integer-spaced data, this is exact.
fn neumaier_smooth(points: &[(f64, f64)], x: f64, w: f64, n: usize) -> f64 {
    let lo = x - w;
    let two_w = 2.0 * w;
    let mut sum: f64 = 0.0;
    let mut comp: f64 = 0.0;
    for i in 0..n {
        let frac = (2 * i + 1) as f64 / (2 * n) as f64;
        let t = lo + two_w * frac;
        let val = eval_piecewise_linear(points, t);
        let t_sum = sum + val;
        if sum.abs() >= val.abs() {
            comp += (sum - t_sum) + val;
        } else {
            comp += (val - t_sum) + sum;
        }
        sum = t_sum;
    }
    (sum + comp) / n as f64
}

fn get_isotonic_xy<T: Float, W: Weight>(points: &[Point<T, T, W>]) -> Vec<(f64, f64)> {
    let regression = IsotonicRegression::new_ascending(points);
    let mut iso_points = regression.into_points();
    iso_points.sort_by(|a, b| a.x().partial_cmp(b.x()).unwrap());
    iso_points
        .iter()
        .map(|p| (p.x().to_f64().unwrap(), p.y().to_f64().unwrap()))
        .collect()
}

// --- f64 tests (tight tolerance — both methods are analytical, differ by 1-2 ULPs) ---

fn check_f64_exact<W: Weight>(points: &[Point<f64, f64, W>], window: f64, label: &str) {
    let iso_xy = get_isotonic_xy(points);
    let smooth = SmoothRegression::new(points, Direction::Ascending, window);

    let x_min = iso_xy.first().unwrap().0;
    let x_max = iso_xy.last().unwrap().0;

    let n_queries = 500;
    let mut max_err: f64 = 0.0;

    for i in 0..=n_queries {
        let x = x_min + (x_max - x_min) * (i as f64 / n_queries as f64);
        let expected = neumaier_smooth(&iso_xy, x, window, 100_000);
        let actual = smooth.interpolate(x).unwrap();
        let err = (actual - expected).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // Both methods compute the exact integral analytically (no approximation),
    // so the only difference is floating-point operation ordering. Should agree
    // to within a few ULPs (~1e-13 for values around 100).
    assert!(
        max_err < 1e-10,
        "[{}] max error {:.2e} — expected near-exact agreement",
        label,
        max_err
    );
}

#[test]
fn test_smooth_exact_f64_linear() {
    let points: Vec<Point<f64>> = (0..100)
        .map(|i| Point::new(i as f64, 2.0 * i as f64 + 1.0))
        .collect();
    check_f64_exact(&points, 5.0, "f64 linear");
}

#[test]
fn test_smooth_exact_f64_noisy() {
    let points: Vec<Point<f64>> = (0..200)
        .map(|i| {
            let x = i as f64;
            let y = x + ((i * 7 + 3) % 13) as f64 - 6.0;
            Point::new(x, y)
        })
        .collect();
    check_f64_exact(&points, 5.0, "f64 noisy");
}

#[test]
fn test_smooth_exact_f64_unit_weight_noisy() {
    let points: Vec<Point<f64, f64, UnitWeight>> = (0..200)
        .map(|i| {
            let x = i as f64;
            let y = x + ((i * 7 + 3) % 13) as f64 - 6.0;
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect();
    check_f64_exact(&points, 5.0, "f64 UnitWeight noisy");
}

#[test]
fn test_smooth_exact_f64_step() {
    let mut points: Vec<Point<f64>> = Vec::new();
    for i in 0..50 {
        points.push(Point::new(i as f64, 0.0));
    }
    for i in 50..100 {
        points.push(Point::new(i as f64, 10.0));
    }
    check_f64_exact(&points, 5.0, "f64 step");
}

// --- f32 tests (approximate — f32 arithmetic in smooth vs f64 reference) ---

fn check_f32_approximate<W: Weight>(
    points: &[Point<f32, f32, W>],
    window: f32,
    tol: f64,
    label: &str,
) {
    let iso_xy = get_isotonic_xy(points);
    let smooth = SmoothRegression::new(points, Direction::Ascending, window);

    let x_min = iso_xy.first().unwrap().0 as f32;
    let x_max = iso_xy.last().unwrap().0 as f32;

    let n_queries = 500;
    let mut max_err: f64 = 0.0;

    for i in 0..=n_queries {
        let x = x_min + (x_max - x_min) * (i as f32 / n_queries as f32);
        let expected = neumaier_smooth(&iso_xy, x as f64, window as f64, 100_000);
        let actual = smooth.interpolate(x).unwrap();
        let err = (actual as f64 - expected).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < tol,
        "[{}] max error {:.2e} exceeds tol {:.2e}",
        label,
        max_err,
        tol
    );
}

#[test]
fn test_smooth_f32_noisy() {
    let points: Vec<Point<f32>> = (0..200)
        .map(|i| {
            let x = i as f32;
            let y = x + ((i * 7 + 3) % 13) as f32 - 6.0;
            Point::new(x, y)
        })
        .collect();
    check_f32_approximate(&points, 5.0f32, 1e-3, "f32 noisy");
}

#[test]
fn test_smooth_f32_unit_weight_noisy() {
    let points: Vec<Point<f32, f32, UnitWeight>> = (0..200)
        .map(|i| {
            let x = i as f32;
            let y = x + ((i * 7 + 3) % 13) as f32 - 6.0;
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect();
    check_f32_approximate(&points, 5.0f32, 1e-3, "f32 UnitWeight noisy");
}

#[test]
fn test_smooth_f32_step() {
    let mut points: Vec<Point<f32>> = Vec::new();
    for i in 0..50 {
        points.push(Point::new(i as f32, 0.0));
    }
    for i in 50..100 {
        points.push(Point::new(i as f32, 10.0));
    }
    check_f32_approximate(&points, 5.0f32, 1e-3, "f32 step");
}
