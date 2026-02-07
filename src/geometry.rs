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
