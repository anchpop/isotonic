use crate::point::{interpolate_two_points, interpolate_x_from_y};
use crate::{Direction, IsotonicRegression, Point, cast};
use eytzinger::SliceExt;
use num_traits::Float;

/// A piecewise linear view of a fit, clamped to the fitted domain.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegressionEvaluator<X, Y = X, W = f64> {
    direction: Direction,
    points: Vec<Point<X, Y, W>>,
}
impl<X: Float, Y: Float, W> RegressionEvaluator<X, Y, W> {
    pub fn new(regression: IsotonicRegression<X, Y, W>) -> Self {
        let direction = regression.direction();
        let mut points = regression.into_points();
        points.eytzingerize(&mut eytzinger::permutation::InplacePermutator);
        Self { direction, points }
    }
    /// Observations in Eytzinger search order, not x order.
    pub fn points(&self) -> &[Point<X, Y, W>] {
        &self.points
    }
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
    pub fn interpolate(&self, x: X) -> Option<Y> {
        self.points.first()?;
        let (lower, upper) = self
            .points
            .eytzinger_interpolative_search_by(|p| p.x().partial_cmp(&x).unwrap());
        Some(match (lower, upper) {
            (Some(a), Some(b)) => interpolate_two_points(&self.points[a], &self.points[b], x),
            (Some(i), None) | (None, Some(i)) => self.points[i].y,
            (None, None) => unreachable!(),
        })
    }

    /// Plateaus invert to their midpoint; values outside the range clamp to an endpoint.
    pub fn invert(&self, at_y: Y) -> Option<X> {
        if self.points.is_empty() {
            return None;
        }
        let mut sorted_points: Vec<_> = self.points.iter().collect();
        sorted_points.sort_by(|a, b| a.x().partial_cmp(b.x()).unwrap());
        let pos = match self.direction {
            Direction::Ascending => {
                sorted_points.binary_search_by(|p| p.y().partial_cmp(&at_y).unwrap())
            }
            Direction::Descending => {
                sorted_points.binary_search_by(|p| at_y.partial_cmp(p.y()).unwrap())
            }
        };
        match pos {
            Ok(exact_idx) => {
                let y_value = sorted_points[exact_idx].y();
                let mut start_idx = exact_idx;
                while start_idx > 0 && sorted_points[start_idx - 1].y() == y_value {
                    start_idx -= 1;
                }
                let mut end_idx = exact_idx;
                while end_idx < sorted_points.len() - 1 && sorted_points[end_idx + 1].y() == y_value
                {
                    end_idx += 1;
                }
                if start_idx < end_idx {
                    let start_x = *sorted_points[start_idx].x();
                    let end_x = *sorted_points[end_idx].x();
                    Some((start_x + end_x) / cast::<f64, X>(2.0))
                } else {
                    Some(*sorted_points[exact_idx].x())
                }
            }
            Err(0) => Some(sorted_points[0].x),
            Err(n) if n == sorted_points.len() => Some(sorted_points[n - 1].x),
            Err(n) => Some(interpolate_x_from_y(
                sorted_points[n - 1],
                sorted_points[n],
                at_y,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnitWeight;
    #[test]
    fn test_invert_ascending() {
        let data = [
            (2.0_f64, 1.0_f64),
            (5.0, 7.0),
            (8.0, 3.0),
            (14.0, 9.0),
            (20.0, 12.0),
        ];
        let points: Vec<Point<f64>> = data.into_iter().map(Point::from).collect();
        let query = RegressionEvaluator::new(IsotonicRegression::new_ascending(&points));
        assert_eq!(query.invert(1.0), Some(2.0));
        assert_eq!(query.invert(5.0), Some(6.5));
        let x = 11.0;
        assert_eq!(query.invert(query.interpolate(x).unwrap()), Some(x));
    }

    #[test]
    fn test_invert_descending() {
        let points = [
            Point::new(-8.0_f64, 16.0_f64),
            Point::new(-1.0, 7.0),
            Point::new(3.0, 9.0),
            Point::new(6.0, 2.0),
            Point::new(15.0, -4.0),
        ];
        let query = RegressionEvaluator::new(IsotonicRegression::new_descending(&points));
        assert_eq!(query.invert(16.0), Some(-8.0));
        assert_eq!(query.invert(-4.0), Some(15.0));
        assert_eq!(query.invert(8.0), Some(1.0));
        let x = 10.5;
        assert_eq!(query.invert(query.interpolate(x).unwrap()), Some(x));
    }

    #[test]
    fn test_invert_single_point() {
        let points = vec![Point::new(1.0, 2.0)];
        let regression = IsotonicRegression::new_ascending(&points);
        let evaluator = RegressionEvaluator::new(regression);
        let x = evaluator.invert(2.0).unwrap();
        assert_eq!(x, 1.0);
    }

    #[test]
    fn test_unit_weight() {
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
        let query = RegressionEvaluator::new(IsotonicRegression::new_ascending(&points));
        assert_eq!(query.interpolate(7.0), Some(5.0));
        assert_eq!(query.invert(5.0), Some(6.5));
    }
}
