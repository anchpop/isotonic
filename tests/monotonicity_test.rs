use isotonic::isotonic_regression::Direction;
use isotonic::weight::Weight;
use isotonic::{IsotonicRegression, Point, SmoothRegression, UnitWeight};
use num_traits::Float;

fn check_isotonic_monotonic<T: Float + std::fmt::Display, W: Weight>(
    points: &[Point<T, T, W>],
    label: &str,
) {
    let regression = IsotonicRegression::new_ascending(points);
    let mut sorted = regression.into_points();
    sorted.sort_by(|a, b| a.x().partial_cmp(b.x()).unwrap());

    let mut violations = 0;
    for i in 1..sorted.len() {
        if *sorted[i].y() < *sorted[i - 1].y() {
            violations += 1;
            if violations <= 5 {
                eprintln!(
                    "[{}] Violation at index {}: x={}, y={} < prev y={}",
                    label,
                    i,
                    sorted[i].x(),
                    sorted[i].y(),
                    sorted[i - 1].y()
                );
            }
        }
    }
    assert_eq!(
        violations, 0,
        "Found {} monotonicity violations in isotonic regression ({})",
        violations, label
    );
}

fn check_smooth_monotonic<T: Float + std::fmt::Display, W: Weight>(
    points: &[Point<T, T, W>],
    window: T,
    label: &str,
) {
    let smooth = SmoothRegression::new(points, Direction::Ascending, window);

    let x_min = *points
        .iter()
        .min_by(|a, b| a.x().partial_cmp(b.x()).unwrap())
        .unwrap()
        .x();
    let x_max = *points
        .iter()
        .max_by(|a, b| a.x().partial_cmp(b.x()).unwrap())
        .unwrap()
        .x();

    let mut prev_y = T::from(f64::NEG_INFINITY).unwrap();
    let mut violations = 0;
    let steps = 2000;
    for i in 0..=steps {
        let frac = T::from(i as f64 / steps as f64).unwrap();
        let x = x_min + (x_max - x_min) * frac;
        if let Some(y) = smooth.interpolate(x) {
            if y < prev_y {
                violations += 1;
                if violations <= 5 {
                    eprintln!(
                        "[{}] Smooth violation at x={}: y={} < prev y={}",
                        label, x, y, prev_y
                    );
                }
            }
            prev_y = y;
        }
    }
    assert_eq!(
        violations, 0,
        "Found {} monotonicity violations in smooth regression ({})",
        violations, label
    );
}

// --- Isotonic tests ---

#[test]
fn test_isotonic_f32_unit_weight_monotonic() {
    let n = 1000;
    let points: Vec<Point<f32, f32, UnitWeight>> = (0..n)
        .map(|i| {
            let x = i as f32;
            let y = x + ((i * 7 + 3) % 13) as f32 - 6.0;
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect();
    check_isotonic_monotonic(&points, "f32 + UnitWeight");
}

#[test]
fn test_isotonic_f64_unit_weight_monotonic() {
    let n = 1000;
    let points: Vec<Point<f64, f64, UnitWeight>> = (0..n)
        .map(|i| {
            let x = i as f64;
            let y = x + ((i * 7 + 3) % 13) as f64 - 6.0;
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect();
    check_isotonic_monotonic(&points, "f64 + UnitWeight");
}

#[test]
fn test_isotonic_f32_f64_weight_monotonic() {
    let n = 1000;
    let points: Vec<Point<f32>> = (0..n)
        .map(|i| {
            let x = i as f32;
            let y = x + ((i * 7 + 3) % 13) as f32 - 6.0;
            Point::new(x, y)
        })
        .collect();
    check_isotonic_monotonic(&points, "f32 + f64 weight");
}

#[test]
fn test_isotonic_f64_f64_weight_monotonic() {
    let n = 1000;
    let points: Vec<Point<f64>> = (0..n)
        .map(|i| {
            let x = i as f64;
            let y = x + ((i * 7 + 3) % 13) as f64 - 6.0;
            Point::new(x, y)
        })
        .collect();
    check_isotonic_monotonic(&points, "f64 + f64 weight");
}

// --- Smooth tests ---

#[test]
fn test_smooth_f32_unit_weight_monotonic() {
    let n = 200;
    let points: Vec<Point<f32, f32, UnitWeight>> = (0..n)
        .map(|i| {
            let x = i as f32;
            let y = x + ((i * 7 + 3) % 13) as f32 - 6.0;
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect();
    check_smooth_monotonic(&points, 5.0f32, "f32 + UnitWeight");
}

#[test]
fn test_smooth_f64_unit_weight_monotonic() {
    let n = 200;
    let points: Vec<Point<f64, f64, UnitWeight>> = (0..n)
        .map(|i| {
            let x = i as f64;
            let y = x + ((i * 7 + 3) % 13) as f64 - 6.0;
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect();
    check_smooth_monotonic(&points, 5.0f64, "f64 + UnitWeight");
}

#[test]
fn test_smooth_f32_f64_weight_monotonic() {
    let n = 200;
    let points: Vec<Point<f32>> = (0..n)
        .map(|i| {
            let x = i as f32;
            let y = x + ((i * 7 + 3) % 13) as f32 - 6.0;
            Point::new(x, y)
        })
        .collect();
    check_smooth_monotonic(&points, 5.0f32, "f32 + f64 weight");
}

#[test]
fn test_smooth_f64_f64_weight_monotonic() {
    let n = 200;
    let points: Vec<Point<f64>> = (0..n)
        .map(|i| {
            let x = i as f64;
            let y = x + ((i * 7 + 3) % 13) as f64 - 6.0;
            Point::new(x, y)
        })
        .collect();
    check_smooth_monotonic(&points, 5.0f64, "f64 + f64 weight");
}
