# isotonic

Isotonic regression adjusts observed y values as little as possible (by weighted
squared error) while making them increase or decrease consistently with x.
This crate fits that trend, evaluates the fitted line, and optionally smooths it.

## Fit

```rust
use isotonic::{IsotonicRegression, Point};
let fit = IsotonicRegression::new_ascending(&[
    Point::new(1_usize, 4.0_f32), Point::new(2, 2.0),
]);
assert_eq!(*fit.points()[0].y(), 3.0);
```

## Query

```rust
use isotonic::{IsotonicRegression, Point, RegressionEvaluator};
let fit = IsotonicRegression::new_ascending(&[
    Point::new(0.0_f64, 2.0_f32), Point::new(4.0, 6.0),
]);
let query = RegressionEvaluator::new(fit);
assert_eq!(query.interpolate(1.0), Some(3.0));
assert_eq!(query.invert(3.0), Some(1.0));
```

## Smooth

```rust
use isotonic::{IsotonicRegression, Point, SmoothRegression};

let fit = IsotonicRegression::new_ascending(&[
    Point::new_with_weight(0.0_f64, 0.0_f32, 2.0),
    Point::new_with_weight(1.0, 1.0, 1.0),
    Point::new_with_weight(2.0, 0.5, 1.0),
]);
let smooth = SmoothRegression::from_regression(fit, 0.25);
let prediction: Option<f32> = smooth.interpolate(1.5_f64);
```

`IsotonicRegression<X, Y, W>` only needs comparable, copyable x coordinates
and floating-point responses. `usize` domains work without converting to floats.
Use `Direction::Descending` for decreasing fits, and the `_sorted` constructors
when observations are already ordered by x. Incomparable x values panic.
Responses and weights should be finite; weights must be positive.

`RegressionEvaluator` offers clamped piecewise-linear interpolation and inversion;
plateaus invert to their midpoint. `SmoothRegression` integrates a box filter over
the fitted line, with constant extension at either end. These query types require
floating-point x coordinates, but x and y may use different precisions.

Weights default to `f64`. `UnitWeight` is a zero-sized alternative for observations
with equal influence. Enable `serde` for serialization and deserialization of
public data types. It is disabled by default.

Licensed under MIT.
