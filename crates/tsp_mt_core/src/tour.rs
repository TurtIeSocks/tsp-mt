use tsp_mt_derive::New;

use crate::LKHNode;

#[derive(Debug, Default, New)]
pub struct Tour {
    pub nodes: Vec<LKHNode>,
}

impl Tour {
    pub fn tour_metrics(&self, threshold_factor: f64) -> TourMetrics {
        let n = self.n();

        if n < 2 {
            log::info!("metrics: n < 2 so there's nothing to report");
            return TourMetrics::default();
        }

        let distances: Vec<f64> = (0..n)
            .map(|i| self.nodes[i].dist(&self.nodes[(i + 1) % n]))
            .collect();
        let total = distances.iter().sum();
        let average = total / (n as f64);
        let threshold = average * threshold_factor;
        let outliers = distances.iter().filter(|d| **d > threshold).count();
        let longest = distances.iter().copied().fold(0.0_f64, f64::max);

        log::info!(
            "metrics: n={n} total_m={total:.0} longest_m={longest:.0} avg_m={average:.0} spike_threshold_m={threshold:.0} spikes={outliers}",
        );

        TourMetrics {
            longest,
            outliers,
            total,
            average,
            threshold,
        }
    }

    fn n(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug, Default, New)]
pub struct TourMetrics {
    pub longest: f64,
    pub outliers: usize,
    pub total: f64,
    pub average: f64,
    pub threshold: f64,
}
