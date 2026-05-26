/// Planar coordinate. Caller is responsible for any projection from
/// lat/lng to a planar coordinate system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}
