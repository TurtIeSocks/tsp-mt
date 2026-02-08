use std::fmt;

const R: f64 = 6_371_000.0;
const NINETY: f64 = 90.0;
const ONE_EIGHTY: f64 = NINETY * 2.0;

/// LKH point representation.
/// For geographic input, `x=lng` and `y=lat` in degrees.
/// For projected internal coordinates, `x/y` are planar values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LKHNode {
    pub x: f64,
    pub y: f64,
}

impl LKHNode {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn from_lat_lng(lat: f64, lng: f64) -> Self {
        Self { x: lng, y: lat }
    }
    pub fn dist(self, rhs: &Self) -> f64 {
        // Haversine meters
        let (lat1, lat2) = (self.y.to_radians(), rhs.y.to_radians());
        let dlat = (rhs.y - self.y).to_radians();
        let dlng = (rhs.x - self.x).to_radians();
        let s1 = (dlat / 2.0).sin();
        let s2 = (dlng / 2.0).sin();
        let h = s1 * s1 + lat1.cos() * lat2.cos() * s2 * s2;
        2.0 * R * h.sqrt().asin()
    }

    pub(crate) fn lat(&self) -> f64 {
        self.y
    }
    pub(crate) fn lng(&self) -> f64 {
        self.x
    }
    pub(crate) fn is_valid(self) -> bool {
        self.y.is_finite()
            && self.x.is_finite()
            && (-NINETY..=NINETY).contains(&self.y)
            && (-ONE_EIGHTY..=ONE_EIGHTY).contains(&self.x)
    }
}

impl fmt::Display for LKHNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b1 = ryu::Buffer::new();
        let mut b2 = ryu::Buffer::new();
        write!(f, "{},{}", b1.format(self.y), b2.format(self.x))
    }
}

#[cfg(test)]
mod tests {
    use super::LKHNode;

    #[test]
    fn new_stores_lat_in_y_and_lng_in_x() {
        let node = LKHNode::from_lat_lng(12.5, -33.75);
        assert_eq!(node.y, 12.5);
        assert_eq!(node.x, -33.75);
    }

    #[test]
    fn valid_bounds_are_accepted() {
        assert!(LKHNode::from_lat_lng(-90.0, -180.0).is_valid());
        assert!(LKHNode::from_lat_lng(90.0, 180.0).is_valid());
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(!LKHNode::from_lat_lng(91.0, 0.0).is_valid());
        assert!(!LKHNode::from_lat_lng(0.0, 181.0).is_valid());
        assert!(!LKHNode::from_lat_lng(f64::NAN, 0.0).is_valid());
        assert!(!LKHNode::from_lat_lng(0.0, f64::INFINITY).is_valid());
    }

    #[test]
    fn dist_is_symmetric_and_zero_for_same_point() {
        let a = LKHNode::from_lat_lng(37.7749, -122.4194);
        let b = LKHNode::from_lat_lng(34.0522, -118.2437);

        let dab = a.dist(&b);
        let dba = b.dist(&a);
        let daa = a.dist(&a);

        assert!((dab - dba).abs() < 1e-6);
        assert!(daa.abs() < 1e-12);
    }

    #[test]
    fn display_formats_as_lat_lng() {
        let node = LKHNode::from_lat_lng(1.5, -2.25);
        assert_eq!(node.to_string(), "1.5,-2.25");
    }
}
