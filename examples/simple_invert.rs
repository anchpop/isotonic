use isotonic::{IsotonicRegression, Point, RegressionEvaluator};

fn main() {
    // Simple ascending example
    let points = vec![
        Point::new(0.0_f64, 0.0),
        Point::new(1.0, 1.0),
        Point::new(2.0, 2.0),
        Point::new(3.0, 3.0),
    ];

    let regression = IsotonicRegression::new_ascending(&points);
    let evaluator = RegressionEvaluator::new(regression);

    println!("Simple linear regression:");
    println!("========================");
    println!();

    // Forward: given x, find y
    println!("Forward (interpolate):");
    let x = 1.5;
    let y = evaluator.interpolate(x).unwrap();
    println!("  x = {} -> y = {}", x, y);
    println!();

    // Inverse: given y, find x
    println!("Inverse (invert):");
    let y = 2.5;
    let x = evaluator.invert(y).unwrap();
    println!("  y = {} -> x = {}", y, x);
    println!();

    // Round trip
    println!("Round trip:");
    let original_x = 1.7;
    let y = evaluator.interpolate(original_x).unwrap();
    let recovered_x = evaluator.invert(y).unwrap();
    println!("  x = {} -> y = {} -> x = {}", original_x, y, recovered_x);
    println!(
        "  Perfect match: {}",
        (original_x - recovered_x).abs() < 0.0001
    );
}
