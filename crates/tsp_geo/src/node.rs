use core::fmt;

use crate::fmath;

const R: f64 = 6_371_000.0;
const NINETY: f64 = 90.0;
const ONE_EIGHTY: f64 = NINETY * 2.0;

/// Geographic point: `x=lng` and `y=lat` in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub x: f64,
    pub y: f64,
}

impl GeoPoint {
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
        let s1 = fmath::sin(dlat / 2.0);
        let s2 = fmath::sin(dlng / 2.0);
        let h = s1 * s1 + fmath::cos(lat1) * fmath::cos(lat2) * s2 * s2;
        2.0 * R * fmath::asin(fmath::sqrt(h))
    }

    /// Embed on a sphere of Earth radius, in meters. The 3D Euclidean
    /// (chord) distance between embedded points is monotone in great-circle
    /// distance and indistinguishable from it at routing scales, so the
    /// solver can optimize plain Euclidean distance in 3D with no
    /// projection distortion anywhere on the globe (poles, date line, ...).
    pub fn unit_sphere_meters(&self) -> [f64; 3] {
        let lat = self.y.to_radians();
        let lng = self.x.to_radians();
        let (sin_lat, cos_lat) = fmath::sin_cos(lat);
        let (sin_lng, cos_lng) = fmath::sin_cos(lng);
        [R * cos_lat * cos_lng, R * cos_lat * sin_lng, R * sin_lat]
    }

    /// Coordinates are finite and within lat/lng bounds.
    pub fn is_valid(self) -> bool {
        self.y.is_finite()
            && self.x.is_finite()
            && (-NINETY..=NINETY).contains(&self.y)
            && (-ONE_EIGHTY..=ONE_EIGHTY).contains(&self.x)
    }
}

impl fmt::Display for GeoPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b1 = ryu::Buffer::new();
        let mut b2 = ryu::Buffer::new();
        write!(f, "{},{}", b1.format(self.y), b2.format(self.x))
    }
}

#[cfg(test)]
mod tests {
    use super::GeoPoint;

    #[test]
    fn new_stores_lat_in_y_and_lng_in_x() {
        let node = GeoPoint::from_lat_lng(12.5, -33.75);
        assert_eq!(node.y, 12.5);
        assert_eq!(node.x, -33.75);
    }

    #[test]
    fn valid_bounds_are_accepted() {
        assert!(GeoPoint::from_lat_lng(-90.0, -180.0).is_valid());
        assert!(GeoPoint::from_lat_lng(90.0, 180.0).is_valid());
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(!GeoPoint::from_lat_lng(91.0, 0.0).is_valid());
        assert!(!GeoPoint::from_lat_lng(0.0, 181.0).is_valid());
        assert!(!GeoPoint::from_lat_lng(f64::NAN, 0.0).is_valid());
        assert!(!GeoPoint::from_lat_lng(0.0, f64::INFINITY).is_valid());
    }

    #[test]
    fn dist_is_symmetric_and_zero_for_same_point() {
        let a = GeoPoint::from_lat_lng(37.7749, -122.4194);
        let b = GeoPoint::from_lat_lng(34.0522, -118.2437);

        let dab = a.dist(&b);
        let dba = b.dist(&a);
        let daa = a.dist(&a);

        assert!((dab - dba).abs() < 1e-6);
        assert!(daa.abs() < 1e-12);
    }

    #[test]
    fn display_formats_as_lat_lng() {
        let node = GeoPoint::from_lat_lng(1.5, -2.25);
        assert_eq!(node.to_string(), "1.5,-2.25");
    }

    #[test]
    fn chord_distance_approximates_haversine_at_routing_scales() {
        let a = GeoPoint::from_lat_lng(52.5200, 13.4050);
        let b = GeoPoint::from_lat_lng(52.5300, 13.4200);
        let chord = {
            let pa = a.unit_sphere_meters();
            let pb = b.unit_sphere_meters();
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt()
        };
        let hav = a.dist(&b);
        assert!((chord - hav).abs() < 0.01, "chord {chord} vs hav {hav}");
    }

    #[test]
    fn embedding_handles_poles_and_dateline() {
        for p in [
            GeoPoint::from_lat_lng(90.0, 0.0),
            GeoPoint::from_lat_lng(-90.0, 45.0),
            GeoPoint::from_lat_lng(0.0, 180.0),
            GeoPoint::from_lat_lng(0.0, -180.0),
        ] {
            let xyz = p.unit_sphere_meters();
            assert!(xyz.iter().all(|c| c.is_finite()));
            let r = (xyz[0].powi(2) + xyz[1].powi(2) + xyz[2].powi(2)).sqrt();
            assert!((r - 6_371_000.0).abs() < 1e-3);
        }
    }
}
