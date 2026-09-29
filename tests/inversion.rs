use isotonic::{Direction, IsotonicRegression, Point, RegressionEvaluator, SmoothRegression};
use num_traits::Float;

#[test]
fn pooled_flat_bottom() {
    for direction in [Direction::Ascending, Direction::Descending] {
        let sign = if direction == Direction::Ascending {
            1.0
        } else {
            -1.0
        };
        let points: Vec<_> = [1.0, -1.0, 0.0, 0.0, 1.0, 2.0]
            .into_iter()
            .enumerate()
            .map(|(i, y)| Point::new(10.0 + i as f64, sign * y))
            .collect();
        let fit = IsotonicRegression::new(&points, direction);
        let evaluator = RegressionEvaluator::new(fit.clone());
        let smooth = SmoothRegression::from_regression(fit, 0.25);
        assert_eq!(evaluator.invert(0.0), Some(11.5));
        assert_eq!(smooth.invert(0.0), Some(11.375));
        for x in [evaluator.invert(sign * 0.01), smooth.invert(sign * 0.01)] {
            let x = x.unwrap();
            assert!(x.is_finite() && (10.0..=15.0).contains(&x), "{x}");
        }
        assert!(
            (smooth
                .interpolate(smooth.invert(sign * 0.01).unwrap())
                .unwrap()
                - sign * 0.01)
                .abs()
                < 1e-12
        );
        for (y, x) in [(-sign, 10.0), (3.0 * sign, 15.0)] {
            assert_eq!(evaluator.invert(y), Some(x));
            assert_eq!(smooth.invert(y), Some(x));
        }
    }
}

fn round_trips<Y: Float>(tolerance: f64) {
    for direction in [Direction::Ascending, Direction::Descending] {
        let sign = if direction == Direction::Ascending {
            1.0
        } else {
            -1.0
        };
        let points: Vec<_> = [0.0, 0.5, 2.0, 2.5, 5.0, 8.0]
            .into_iter()
            .enumerate()
            .map(|(i, y)| Point::new(10.0 + i as f64, Y::from(sign * y).unwrap()))
            .collect();
        let fit = IsotonicRegression::new(&points, direction);
        let evaluator = RegressionEvaluator::new(fit.clone());
        for smooth in [
            SmoothRegression::from_regression(fit.clone(), 0.25),
            SmoothRegression::new(&points, direction, 0.25),
            SmoothRegression::new_sorted(&points, direction, 0.25),
        ] {
            for i in 0..=1000 {
                let x = 10.0 + i as f64 / 200.0;
                for recovered in [
                    evaluator.invert(evaluator.interpolate(x).unwrap()).unwrap(),
                    smooth.invert(smooth.interpolate(x).unwrap()).unwrap(),
                ] {
                    assert!(
                        (recovered - x).abs() < tolerance,
                        "{direction:?}: {x} -> {recovered}"
                    );
                }
            }
        }
    }
}

#[test]
fn round_trips_f64() {
    round_trips::<f64>(1e-10);
}

#[test]
fn round_trips_mixed_precision() {
    round_trips::<f32>(2e-5);
}

#[test]
fn empty_singleton_and_constant() {
    for direction in [Direction::Ascending, Direction::Descending] {
        for n in [0, 1, 17] {
            let points: Vec<_> = (0..n).map(|i| Point::new(10.0 + i as f64, 2.0)).collect();
            let fit = IsotonicRegression::new(&points, direction);
            let evaluator = RegressionEvaluator::new(fit.clone());
            let smooth = SmoothRegression::from_regression(fit, 0.25);
            for y in [1.0, 2.0, 3.0] {
                let expected = if n == 0 {
                    None
                } else if y == 2.0 {
                    Some(10.0 + (n - 1) as f64 / 2.0)
                } else if (y < 2.0) == (direction == Direction::Ascending) {
                    Some(10.0)
                } else {
                    Some(10.0 + (n - 1) as f64)
                };
                assert_eq!(evaluator.invert(y), expected);
                assert_eq!(smooth.invert(y), expected);
            }
        }
    }
}

#[test]
fn tiny_curvature_is_not_flat() {
    let points = [
        Point::new(10.0, 0.0),
        Point::new(11.0, 0.0),
        Point::new(12.0, 1e-15),
    ];
    let smooth = SmoothRegression::new(&points, Direction::Ascending, 0.25);
    for x in [10.8, 10.9, 11.0, 11.1, 11.2] {
        let y = smooth.interpolate(x).unwrap();
        let recovered = smooth.invert(y).unwrap();
        assert!((recovered - x).abs() < 1e-12, "{x} -> {recovered}");
    }
}

#[test]
fn interior_plateau_edges() {
    for direction in [Direction::Ascending, Direction::Descending] {
        let sign = if direction == Direction::Ascending {
            1.0
        } else {
            -1.0
        };
        let points: Vec<_> = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0]
            .into_iter()
            .enumerate()
            .map(|(i, y)| Point::new(10.0 + i as f64, sign * y))
            .collect();
        let fit = IsotonicRegression::new(&points, direction);
        let evaluator = RegressionEvaluator::new(fit.clone());
        let smooth = SmoothRegression::from_regression(fit, 0.25);
        assert_eq!(evaluator.invert(sign), Some(12.5));
        assert_eq!(smooth.invert(sign), Some(12.5));
        for y in [0.99 * sign, 1.01 * sign] {
            let x = smooth.invert(y).unwrap();
            assert!((smooth.interpolate(x).unwrap() - y).abs() < 1e-12);
        }
    }
}

#[test]
fn tiny_responses_round_trip() {
    let points: Vec<_> = (0..6)
        .map(|i| Point::new(10.0 + i as f32, 1e-24 * i as f32))
        .collect();
    let smooth = SmoothRegression::new(&points, Direction::Ascending, 0.25);
    for x in [10.4_f32, 11.3, 12.5, 14.1] {
        let back = smooth.invert(smooth.interpolate(x).unwrap()).unwrap();
        assert!((back - x).abs() < 1e-3, "{x} -> {back}");
    }
}
