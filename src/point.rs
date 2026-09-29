use crate::{Weight, cast};
use num_traits::Float;

/// One observation, with independent domain, response, and weight types.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point<X, Y = X, W = f64> {
    pub(crate) x: X,
    pub(crate) y: Y,
    pub(crate) weight: W,
}

impl<X, Y> Point<X, Y> {
    pub fn new(x: X, y: Y) -> Self {
        Self::new_with_weight(x, y, 1.0)
    }
}
impl<X, Y, W> Point<X, Y, W> {
    pub fn new_with_weight(x: X, y: Y, weight: W) -> Self {
        Self { x, y, weight }
    }
    pub fn x(&self) -> &X {
        &self.x
    }
    pub fn y(&self) -> &Y {
        &self.y
    }
    pub fn weight(&self) -> &W {
        &self.weight
    }
}
impl<X, Y, W: Weight> From<(X, Y)> for Point<X, Y, W> {
    fn from((x, y): (X, Y)) -> Self {
        Self::new_with_weight(x, y, W::unit())
    }
}

/// Evaluate the line joining two observations. Their x coordinates must differ.
pub fn interpolate_two_points<X: Float, Y: Float, W>(
    a: &Point<X, Y, W>,
    b: &Point<X, Y, W>,
    x: X,
) -> Y {
    let fraction: Y = cast((x - a.x) / (b.x - a.x));
    fraction.mul_add(b.y - a.y, a.y)
}

/// Locate a response on the line joining two observations with different y values.
pub fn interpolate_x_from_y<X: Float, Y: Float, W>(
    a: &Point<X, Y, W>,
    b: &Point<X, Y, W>,
    at_y: Y,
) -> X {
    let fraction: X = cast((at_y - a.y) / (b.y - a.y));
    fraction.mul_add(b.x - a.x, a.x)
}
