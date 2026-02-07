use std::fmt;

const EARTH_RADIUS_METERS: f64 = 6_371_000.0;

/// Geographic point (degrees).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub lat: f64,
    pub lng: f64,
}

impl Point {
    pub fn dist(self, rhs: &Point) -> f64 {
        let (lat1, lat2) = (self.lat.to_radians(), rhs.lat.to_radians());
        let dlat = (rhs.lat - self.lat).to_radians();
        let dlng = (rhs.lng - self.lng).to_radians();
        let s1 = (dlat / 2.0).sin();
        let s2 = (dlng / 2.0).sin();
        let h = s1 * s1 + lat1.cos() * lat2.cos() * s2 * s2;
        2.0 * EARTH_RADIUS_METERS * h.sqrt().asin()
    }

    pub fn is_valid(self) -> bool {
        self.lat.is_finite()
            && self.lng.is_finite()
            && (-90.0..=90.0).contains(&self.lat)
            && (-180.0..=180.0).contains(&self.lng)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b1 = ryu::Buffer::new();
        let mut b2 = ryu::Buffer::new();
        write!(f, "{},{}", b1.format(self.lat), b2.format(self.lng))
    }
}
