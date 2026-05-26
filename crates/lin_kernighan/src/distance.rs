use crate::coord::Point2D;

/// TSPLIB EUC_2D edge weight. Matches LKH's `Distance_EUC_2D`:
/// `(int)(Scale * sqrt(dx² + dy²) + 0.5)` with `Scale = 1`.
///
/// LKH uses integer edge weights pervasively (cost matrix, Pi values,
/// gain accounting, candidate alpha values), so we preserve that here
/// by returning `i64` rather than `f64`. Callers project to a planar
/// coordinate system and scale by an integer factor before constructing
/// the [`Problem`](crate::Problem); the rounding here is the final nint.
#[inline]
pub fn euc_2d(a: Point2D, b: Point2D) -> i64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let d = (dx * dx + dy * dy).sqrt();
    (d + 0.5) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euc_2d_matches_tsplib_nint() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert_eq!(euc_2d(a, b), 5);
        assert_eq!(euc_2d(b, a), 5);
    }

    #[test]
    fn euc_2d_is_zero_for_same_point() {
        let a = Point2D::new(7.5, -2.25);
        assert_eq!(euc_2d(a, a), 0);
    }

    #[test]
    fn euc_2d_rounds_half_up() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(1.0, 1.0);
        assert_eq!(euc_2d(a, b), 1);

        let c = Point2D::new(0.0, 0.0);
        let d = Point2D::new(0.5, 0.0);
        assert_eq!(euc_2d(c, d), 1);
    }
}
