use isotonic::{IsotonicRegression, Point, RegressionEvaluator, SmoothRegression};

fn main() {
    // Example: Frequency-based measurements with some noise
    let measurements = vec![
        Point::new(100.0_f64, 10.0),
        Point::new(200.0, 25.0),
        Point::new(300.0, 35.0),
        Point::new(400.0, 42.0),
        Point::new(500.0, 55.0),
        Point::new(600.0, 58.0),
        Point::new(700.0, 70.0),
        Point::new(800.0, 75.0),
        Point::new(900.0, 85.0),
        Point::new(1000.0, 95.0),
    ];

    println!("Building isotonic regression...");
    let regression = IsotonicRegression::new_ascending(&measurements);
    let evaluator = RegressionEvaluator::new(regression.clone());

    println!("Creating smoothed regression with window=20...");
    let smooth = SmoothRegression::from_regression(regression, 20.0);

    println!("\nComparing approaches for frequency=550:");
    let base_freq = 550.0;

    // Old approach: smooth at query time
    println!("\nOld approach (smooth at query time):");
    let lower_freq = base_freq * 0.8;
    let upper_freq = base_freq * 1.2;
    let predictions = [
        evaluator.interpolate(lower_freq).unwrap(),
        evaluator.interpolate(base_freq).unwrap(),
        evaluator.interpolate(upper_freq).unwrap(),
    ];
    let avg = (predictions[0] + predictions[1] + predictions[2]) / 3.0;
    println!("  Lower ({}): {:.2}", lower_freq, predictions[0]);
    println!("  Base  ({}): {:.2}", base_freq, predictions[1]);
    println!("  Upper ({}): {:.2}", upper_freq, predictions[2]);
    println!("  Average: {:.2}", avg);

    // New approach: pre-smoothed
    println!("\nNew approach (pre-smoothed):");
    let smooth_value = smooth.interpolate(base_freq).unwrap();
    println!("  Direct lookup at {}: {:.2}", base_freq, smooth_value);

    println!("\n\nPerformance comparison:");
    println!("======================================");
    println!("Old: 3 interpolations per query");
    println!("New: 1 interpolation per query (3x faster!)");

    println!("\n\nInversion example:");
    println!("==================");
    let target_value = 60.0;
    let freq = smooth.invert(target_value).unwrap();
    println!("To get value {}, use frequency {:.2}", target_value, freq);

    // Verify
    let check = smooth.interpolate(freq).unwrap();
    println!("Verification: frequency {:.2} -> value {:.2}", freq, check);

    println!("\n\nTable of smoothed values:");
    println!("=========================");
    println!("Frequency | Smooth Value");
    println!("----------|-------------");
    for i in 1..=10 {
        let freq = i as f64 * 100.0;
        let value = smooth.interpolate(freq).unwrap();
        println!("{:9.0} | {:12.2}", freq, value);
    }
}
