use isotonic::{IsotonicRegression, Point, RegressionEvaluator, SmoothRegression};

fn main() {
    // Create some data with sharp corners
    let points = vec![
        Point::new(0.0_f64, 0.0),
        Point::new(1.0, 1.0),
        Point::new(2.0, 1.0), // Flat section
        Point::new(3.0, 1.0),
        Point::new(4.0, 4.0), // Sharp jump
        Point::new(5.0, 5.0),
    ];

    println!("Smooth Regression Example");
    println!("=========================\n");

    // Build the isotonic regression
    let regression = IsotonicRegression::new_ascending(&points);

    // Create an evaluator for original regression queries
    let evaluator = RegressionEvaluator::new(regression.clone());

    // Create smoothed versions with different window sizes
    let smooth_small = SmoothRegression::from_regression(regression.clone(), 0.2);
    let smooth_medium = SmoothRegression::from_regression(regression.clone(), 0.5);
    let smooth_large = SmoothRegression::from_regression(regression, 1.0);

    println!("Comparing original vs smoothed (different window sizes):\n");
    println!("   x   | Original | w=0.2  | w=0.5  | w=1.0");
    println!("-------|----------|--------|--------|--------");

    for i in 0..=50 {
        let x = i as f64 * 0.1;
        let y_orig = evaluator.interpolate(x).unwrap();
        let y_small = smooth_small.interpolate(x).unwrap();
        let y_medium = smooth_medium.interpolate(x).unwrap();
        let y_large = smooth_large.interpolate(x).unwrap();

        println!(
            " {:5.2} | {:8.2} | {:6.2} | {:6.2} | {:6.2}",
            x, y_orig, y_small, y_medium, y_large
        );
    }

    println!("\n\nInversion with smoothed regression:");
    println!("===================================\n");

    let regression2 = IsotonicRegression::new_ascending(&points);
    let smooth = SmoothRegression::from_regression(regression2, 0.5);

    println!("Finding x values for target y values:");
    for target_y in [0.5, 1.5, 2.5, 3.5, 4.5] {
        let x = smooth.invert(target_y).unwrap();
        let verify_y = smooth.interpolate(x).unwrap();
        println!(
            "  Target y={:.1} -> x={:.2} (verify: y={:.2})",
            target_y, x, verify_y
        );
    }

    println!("\n\nPerformance benefit:");
    println!("====================");
    println!("Instead of smoothing at query time like this:");
    println!("  let y = (interp(x*0.8) + interp(x) + interp(x*1.2)) / 3.0;  // 3 lookups!");
    println!("\nYou can pre-smooth once:");
    println!("  let smooth = SmoothRegression::from_regression(reg, window);");
    println!("  let y = smooth.interpolate(x);  // 1 lookup - 3x faster!");
}
