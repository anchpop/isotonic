use isotonic::isotonic_regression::Direction;
use isotonic::{IsotonicRegression, Point, SmoothRegression, UnitWeight};
use std::time::Instant;

fn generate_points(n: usize) -> Vec<Point<f64, f64, UnitWeight>> {
    let mut x = 0.0_f64;
    let mut seed = 12345u64;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / u32::MAX as f64) * 10.0 - 5.0;
            let y = x + noise;
            let p = Point::new_with_weight(x, y, UnitWeight);
            x += 1.0;
            p
        })
        .collect()
}

fn main() {
    for &n in &[1_000, 10_000, 100_000] {
        println!("\n=== n = {} ===", n);
        let points = generate_points(n);

        // End-to-end timings
        let iters = if n <= 10_000 { 100 } else { 10 };

        let t0 = Instant::now();
        for _ in 0..iters {
            let reg = IsotonicRegression::new_ascending_sorted(&points);
            let _ = SmoothRegression::from_regression(reg, 5.0);
        }
        let dt1 = t0.elapsed() / iters;
        println!("from_regression:  {:?}", dt1);

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = SmoothRegression::new_sorted(&points, Direction::Ascending, 5.0);
        }
        let dt2 = t0.elapsed() / iters;
        println!("new_sorted:       {:?}", dt2);

        // Breakdown: isotonic
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = IsotonicRegression::new_ascending_sorted(&points);
        }
        let dt_iso = t0.elapsed() / iters;
        println!("  isotonic:       {:?}", dt_iso);
        println!("  smooth build:   {:?}", dt1.saturating_sub(dt_iso));

        // Count boundaries to understand segment count
        let reg = IsotonicRegression::new_ascending_sorted(&points);
        let iso_points = reg.points();
        let x_min = *iso_points.first().unwrap().x();
        let x_max = *iso_points.last().unwrap().x();
        let w = 5.0_f64;

        // Current: x-w, x, x+w for each point
        let mut boundaries3 = Vec::new();
        for p in iso_points {
            let x = *p.x();
            boundaries3.push(x - w);
            boundaries3.push(x);
            boundaries3.push(x + w);
        }
        boundaries3.sort_by(|a, b| a.partial_cmp(b).unwrap());
        boundaries3.dedup();
        boundaries3.retain(|&x| x >= x_min && x <= x_max);

        // Without x_i: only x-w, x+w
        let mut boundaries2 = Vec::new();
        for p in iso_points {
            let x = *p.x();
            boundaries2.push(x - w);
            boundaries2.push(x + w);
        }
        boundaries2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        boundaries2.dedup();
        boundaries2.retain(|&x| x >= x_min && x <= x_max);

        println!("  boundaries (x-w,x,x+w): {}", boundaries3.len());
        println!("  boundaries (x-w,x+w):   {}", boundaries2.len());
        println!(
            "  segment reduction:       {:.0}%",
            (1.0 - boundaries2.len() as f64 / boundaries3.len() as f64) * 100.0
        );
    }
}
