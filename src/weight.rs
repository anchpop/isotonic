use crate::cast;
use num_traits::Float;
use std::fmt::Debug;

/// Trait for weight types used in [`crate::Point`].
///
/// This trait abstracts over the weight storage in `Point`, allowing a zero-sized
/// [`UnitWeight`] type for cases where weights are not needed (saving 8 bytes per point).
pub trait Weight: Copy + Clone + Debug + PartialEq {
    /// Returns the unit weight (equivalent to 1.0).
    fn unit() -> Self;
    /// Convert weight directly to a Float type, bypassing f64.
    fn to_coord<T: Float>(&self) -> T;
}

impl Weight for f64 {
    fn unit() -> Self {
        1.0
    }
    fn to_coord<T: Float>(&self) -> T {
        cast(*self)
    }
}

/// A zero-sized weight type that always represents a weight of 1.0.
///
/// Using `Point<T, T, UnitWeight>` instead of `Point<T>` (which defaults to `Point<T, T, f64>`)
/// saves 8 bytes per point, which can improve cache performance for large datasets
/// where weights are not needed.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnitWeight;

impl Weight for UnitWeight {
    fn unit() -> Self {
        UnitWeight
    }
    fn to_coord<T: Float>(&self) -> T {
        T::one()
    }
}
