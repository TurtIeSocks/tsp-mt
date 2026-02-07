use crate::node::LKHNode;

pub(crate) struct TourGeometry;

impl TourGeometry {
    pub(crate) fn tour_length(points: &[LKHNode], tour: &[usize]) -> f64 {
        let n = tour.len();
        let mut sum = 0.0;
        for i in 0..n {
            let a = points[tour[i]];
            let b = points[tour[(i + 1) % n]];
            sum += Self::dist(a, b);
        }
        sum
    }

    #[inline]
    pub(crate) fn dist(a: LKHNode, b: LKHNode) -> f64 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub(crate) fn centroid_of_indices(coords: &[LKHNode], idxs: &[usize]) -> LKHNode {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for &i in idxs {
            sx += coords[i].x;
            sy += coords[i].y;
        }
        let n = idxs.len().max(1) as f64;
        LKHNode::new(sy / n, sx / n)
    }

    pub(crate) fn rotate_cycle(tour: &[usize], start_node: usize) -> Vec<usize> {
        let Some(pos) = tour.iter().position(|&x| x == start_node) else {
            return tour.to_vec();
        };
        let mut out = Vec::with_capacity(tour.len());
        out.extend_from_slice(&tour[pos..]);
        out.extend_from_slice(&tour[..pos]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::TourGeometry;
    use crate::node::LKHNode;

    #[test]
    fn dist_uses_euclidean_metric() {
        let a = LKHNode::new(0.0, 0.0);
        let b = LKHNode::new(4.0, 3.0);
        assert!((TourGeometry::dist(a, b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn tour_length_closes_cycle() {
        let points = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(0.0, 1.0),
            LKHNode::new(1.0, 1.0),
            LKHNode::new(1.0, 0.0),
        ];
        let tour = vec![0, 1, 2, 3];
        let length = TourGeometry::tour_length(&points, &tour);
        assert!((length - 4.0).abs() < 1e-12);
    }

    #[test]
    fn centroid_of_indices_averages_coordinates() {
        let coords = vec![
            LKHNode::new(2.0, 1.0),
            LKHNode::new(4.0, 3.0),
            LKHNode::new(6.0, 5.0),
        ];
        let centroid = TourGeometry::centroid_of_indices(&coords, &[0, 2]);
        assert!((centroid.y - 4.0).abs() < 1e-12);
        assert!((centroid.x - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rotate_cycle_starts_at_requested_node() {
        let rotated = TourGeometry::rotate_cycle(&[10, 20, 30, 40], 30);
        assert_eq!(rotated, vec![30, 40, 10, 20]);
    }

    #[test]
    fn rotate_cycle_returns_original_if_node_missing() {
        let original = vec![1, 2, 3];
        let rotated = TourGeometry::rotate_cycle(&original, 99);
        assert_eq!(rotated, original);
    }
}
