use isotonic::{IsotonicRegression, Point, RegressionEvaluator};

fn main() {
    // Create some example data
    let points = vec![
        Point::new(0.0_f64, 0.0),
        Point::new(1.0, 1.0),
        Point::new(2.0, 4.0),
        Point::new(3.0, 9.0),
        Point::new(4.0, 16.0),
    ];

    let regression = IsotonicRegression::new_ascending(&points);
    let evaluator = RegressionEvaluator::new(regression);

    println!("Isotonic Regression Example");
    println!("===========================\n");

    // Forward: given x, find y
    println!("Forward (interpolate - given x, find y):");
    for x in [0.5, 1.5, 2.5, 3.5] {
        let y = evaluator.interpolate(x).unwrap();
        println!("  x = {:.1} -> y = {:.2}", x, y);
    }
    println!();

    // Inverse: given y, find x
    println!("Inverse (invert - given y, find x):");
    for y in [2.0, 5.0, 10.0, 15.0] {
        let x = evaluator.invert(y).unwrap();
        println!("  y = {:.1} -> x = {:.2}", y, x);
    }
    println!();

    // Round trip demonstration
    println!("Round trip verification:");
    let original_x = 2.7;
    let y = evaluator.interpolate(original_x).unwrap();
    let recovered_x = evaluator.invert(y).unwrap();
    println!("  Start with x = {}", original_x);
    println!("  Interpolate to get y = {:.2}", y);
    println!("  Invert to recover x = {:.2}", recovered_x);
    println!("  Error: {:.6}", (original_x - recovered_x).abs());
}
