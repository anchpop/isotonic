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
