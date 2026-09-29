use isotonic::{Direction, IsotonicRegression, Point, RegressionEvaluator, SmoothRegression};

#[test]
fn integer_domain_weighted_fit() {
    let fit = IsotonicRegression::new_ascending(&[
        Point::new_with_weight(8_usize, 6_f32, 3.0),
        Point::new_with_weight(9, 2.0, 1.0),
    ]);
    assert_eq!(
        fit.points().iter().map(|p| *p.y()).collect::<Vec<_>>(),
        [5.0, 5.0]
    );
    assert_eq!(fit.len(), 2);
}

#[test]
fn mixed_precision_keeps_domain_resolution() {
    let offset = 1e12_f64;
    let points = [
        Point::new(offset, 0_f32),
        Point::new(offset + 2.0, 2.0),
        Point::new(offset + 4.0, 4.0),
    ];
    let smooth = SmoothRegression::new(&points, Direction::Ascending, 0.25);
    let y: f32 = smooth.interpolate(offset + 1.0).unwrap();
    assert!((y - 1.0).abs() < 1e-4);
    let x: f64 = smooth.invert(y).unwrap();
    assert!((x - offset - 1.0).abs() < 1e-3);
}

#[test]
fn descending_and_clamped_queries() {
    let fit = IsotonicRegression::new_descending(&[
        Point::new(2_f64, 9_f32),
        Point::new(4.0, 3.0),
        Point::new(6.0, 5.0),
    ]);
    assert_eq!(
        fit.points().iter().map(|p| *p.y()).collect::<Vec<_>>(),
        [9.0, 4.0, 4.0]
    );
    let query = RegressionEvaluator::new(fit);
    assert_eq!(query.interpolate(-20.0), Some(9.0));
    assert_eq!(query.interpolate(20.0), Some(4.0));
    assert_eq!(query.invert(20.0), Some(2.0));
    assert_eq!(query.invert(-20.0), Some(6.0));
    assert_eq!(query.invert(4.0), Some(5.0));
}

#[test]
fn incomparable_coordinates_always_panic() {
    for sorted in [false, true] {
        for points in [
            vec![Point::new(f64::NAN, 1.0)],
            vec![Point::new(0.0, 0.0), Point::new(f64::NAN, 1.0)],
        ] {
            assert!(
                std::panic::catch_unwind(|| {
                    if sorted {
                        IsotonicRegression::new_ascending_sorted(&points)
                    } else {
                        IsotonicRegression::new_ascending(&points)
                    }
                })
                .is_err()
            );
        }
    }
}

#[test]
fn inverse_normalizes_before_converting_response_precision() {
    let fit =
        IsotonicRegression::new_ascending(&[Point::new(0_f32, 0_f64), Point::new(1.0, 1e-100)]);
    let query = RegressionEvaluator::new(fit);
    assert_eq!(query.invert(5e-101), Some(0.5));
    assert_eq!(query.interpolate(0.5), Some(5e-101));
}

#[cfg(feature = "serde")]
#[test]
fn public_types_support_serde() {
    fn serializable<T: serde::Serialize + for<'a> serde::Deserialize<'a>>() {}
    serializable::<Point<usize, f32>>();
    serializable::<IsotonicRegression<usize, f32>>();
    serializable::<RegressionEvaluator<f64, f32>>();
    serializable::<SmoothRegression<f64, f32>>();
    serializable::<Direction>();
    serializable::<isotonic::UnitWeight>();
}
