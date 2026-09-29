#![doc = include_str!("../README.md")]

pub mod isotonic_regression;
pub mod point;
pub mod regression_evaluator;
pub mod smooth_regression;
pub mod weight;
pub use isotonic_regression::{Direction, IsotonicRegression};
pub use point::Point;
pub use regression_evaluator::RegressionEvaluator;
pub use smooth_regression::SmoothRegression;
pub use weight::{UnitWeight, Weight};

pub(crate) fn cast<A: num_traits::NumCast, B: num_traits::NumCast>(value: A) -> B {
    num_traits::cast(value).expect("numeric conversion failed")
}

/// Return the last boundary at/before the target and the first at/after it
/// in x order. Equal values span a plateau; otherwise these are neighbours.
fn inverse_bounds<T, Y: num_traits::Float>(
    values: &[T],
    direction: Direction,
    target: Y,
    y: impl Fn(&T) -> Y,
) -> (Option<usize>, Option<usize>) {
    use eytzinger::SliceExt;
    use std::cmp::Ordering::{Greater, Less};

    let cmp = |value: &T| match direction {
        Direction::Ascending => y(value).partial_cmp(&target).unwrap(),
        Direction::Descending => target.partial_cmp(&y(value)).unwrap(),
    };
    // The fork returns (last <=, first >). Never returning Equal lets us
    // choose which side owns equality, finding both ends without scanning.
    let (left, _) = values
        .eytzinger_interpolative_search_by(|v| if cmp(v) == Greater { Greater } else { Less });
    let (_, right) =
        values.eytzinger_interpolative_search_by(|v| if cmp(v) == Less { Less } else { Greater });
    (left, right)
}
