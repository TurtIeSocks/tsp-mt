use std::fmt;

/// LKH point representation.
/// For geographic input, `x=lng` and `y=lat` in degrees.
/// For projected internal coordinates, `x/y` are planar values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LKHNode {
    pub x: f64,
    pub y: f64,
}

impl LKHNode {
    pub fn new(lat_or_y: f64, lng_or_x: f64) -> Self {
        Self {
            x: lng_or_x,
            y: lat_or_y,
        }
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
            && (-90.0..=90.0).contains(&self.y)
            && (-180.0..=180.0).contains(&self.x)
    }
}

impl fmt::Display for LKHNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b1 = ryu::Buffer::new();
        let mut b2 = ryu::Buffer::new();
        write!(f, "{},{}", b1.format(self.y), b2.format(self.x))
    }
}
