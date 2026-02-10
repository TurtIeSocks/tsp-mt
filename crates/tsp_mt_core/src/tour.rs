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

#[cfg(test)]
mod tests {
    use super::{Tour, TourMetrics};
    use crate::LKHNode;

    #[test]
    fn tour_metrics_returns_default_for_short_tours() {
        let empty = Tour::new(Vec::new());
        let single = Tour::new(vec![LKHNode::from_lat_lng(0.0, 0.0)]);

        assert_eq!(empty.tour_metrics(2.0).total, TourMetrics::default().total);
        assert_eq!(
            single.tour_metrics(2.0).outliers,
            TourMetrics::default().outliers
        );
        assert_eq!(
            single.tour_metrics(2.0).longest,
            TourMetrics::default().longest
        );
    }

    #[test]
    fn tour_metrics_computes_cycle_lengths_and_outliers() {
        let tour = Tour::new(vec![
            LKHNode::from_lat_lng(0.0, 0.0),
            LKHNode::from_lat_lng(0.0, 1.0),
            LKHNode::from_lat_lng(0.0, 2.0),
        ]);

        let metrics = tour.tour_metrics(1.2);
        let d01 = tour.nodes[0].dist(&tour.nodes[1]);
        let d12 = tour.nodes[1].dist(&tour.nodes[2]);
        let d20 = tour.nodes[2].dist(&tour.nodes[0]);
        let expected_total = d01 + d12 + d20;
        let expected_average = expected_total / 3.0;
        let expected_threshold = expected_average * 1.2;

        assert!((metrics.total - expected_total).abs() < 1e-6);
        assert!((metrics.average - expected_average).abs() < 1e-6);
        assert!((metrics.threshold - expected_threshold).abs() < 1e-6);
        assert!((metrics.longest - d20).abs() < 1e-6);
        assert_eq!(metrics.outliers, 1);
    }
}
