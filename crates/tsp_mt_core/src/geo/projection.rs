use map_3d::{self, Ellipsoid};

use crate::node::LKHNode;

type Precision = f64;
type Geocentric = (Precision, Precision, Precision);
type Topocentric = (Precision, Precision);

pub(crate) struct PlaneProjection {
    center: Geocentric,
    x: Geocentric,
    y: Geocentric,
    z: Geocentric,
    radius: Precision,
    adjusted_radius: Precision,
    points: Vec<Geocentric>,
}

impl Default for PlaneProjection {
    fn default() -> Self {
        PlaneProjection {
            center: (0.0, 0.0, 0.0),
            x: (0.0, 0.0, 0.0),
            y: (0.0, 0.0, 0.0),
            z: (0.0, 0.0, 0.0),
            radius: 0.0,
            adjusted_radius: 0.0,
            points: vec![],
        }
    }
}

impl PlaneProjection {
    pub(crate) fn new(input: &[LKHNode]) -> PlaneProjection {
        let mut plane = PlaneProjection {
            points: input
                .iter()
                .map(|p| {
                    map_3d::geodetic2ecef(
                        p.y.to_radians(),
                        p.x.to_radians(),
                        0.0,
                        Ellipsoid::default(),
                    )
                })
                .collect(),
            ..Default::default()
        };
        let (plane_center_lat, plane_center_lon) = plane.compute_plane_center();

        plane.center = map_3d::geodetic2ecef(
            plane_center_lat,
            plane_center_lon,
            0.0,
            Ellipsoid::default(),
        );
        plane.z = (
            plane_center_lat.cos() * plane_center_lon.cos(),
            plane_center_lat.cos() * plane_center_lon.sin(),
            plane_center_lat.sin(),
        );
        plane.y = plane.normalize((-plane.center.1, plane.center.0, 0.0));
        plane.x = plane.cross_product(plane.z, plane.y);

        plane
    }

    pub(crate) fn radius(mut self, radius: Precision) -> Self {
        self.radius = radius;
        let earth_minor = Ellipsoid::default().parameters().1;
        self.adjusted_radius = 0.5 * earth_minor * (2.0 * self.radius / earth_minor).sin();
        self
    }

    pub(crate) fn project(&self) -> Vec<LKHNode> {
        let global_scale = self.dot_product(self.center, self.z) / self.adjusted_radius;
        let offset_x = self.dot_product(self.center, self.x) / self.adjusted_radius;
        self.points
            .iter()
            .map(|p| {
                let scale = global_scale / self.dot_product(*p, self.z);
                LKHNode::new(
                    self.dot_product(*p, self.y) * scale,
                    self.dot_product(*p, self.x) * scale - offset_x,
                )
            })
            .collect()
    }

    fn euclidean_norm2(&self, x: Geocentric) -> Precision {
        x.0 * x.0 + x.1 * x.1 + x.2 * x.2
    }

    fn dot_product(&self, x: Geocentric, y: Geocentric) -> Precision {
        x.0 * y.0 + x.1 * y.1 + x.2 * y.2
    }

    fn cross_product(&self, x: Geocentric, y: Geocentric) -> Geocentric {
        (
            x.1 * y.2 - x.2 * y.1,
            x.2 * y.0 - x.0 * y.2,
            x.0 * y.1 - x.1 * y.0,
        )
    }

    fn normalize(&self, x: Geocentric) -> Geocentric {
        let l = self.euclidean_norm2(x).sqrt();
        (x.0 / l, x.1 / l, x.2 / l)
    }

    fn radial_project(&self, p: Geocentric) -> Topocentric {
        let t = 1.0 - Ellipsoid::default().parameters().2;
        (
            (p.2 / (t * t * (p.0 * p.0 + p.1 * p.1).sqrt())).atan(),
            p.1.atan2(p.0),
        )
    }

    fn compute_plane_center(&self) -> Topocentric {
        let mut dir = (0.0, 0.0, 0.0);
        for (x, y, z) in &self.points {
            dir.0 += x;
            dir.1 += y;
            dir.2 += z;
        }
        self.radial_project(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::PlaneProjection;
    use crate::node::LKHNode;

    #[test]
    fn project_preserves_point_count_and_returns_finite_coordinates() {
        let input = vec![
            LKHNode::new(37.7749, -122.4194),
            LKHNode::new(34.0522, -118.2437),
            LKHNode::new(40.7128, -74.0060),
        ];

        let projected = PlaneProjection::new(&input).radius(70.0).project();

        assert_eq!(projected.len(), input.len());
        assert!(projected.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    #[test]
    fn project_same_input_is_deterministic() {
        let input = vec![
            LKHNode::new(48.8566, 2.3522),
            LKHNode::new(51.5074, -0.1278),
        ];

        let a = PlaneProjection::new(&input).radius(100.0).project();
        let b = PlaneProjection::new(&input).radius(100.0).project();

        assert_eq!(a.len(), b.len());
        for (lhs, rhs) in a.iter().zip(b.iter()) {
            assert!((lhs.x - rhs.x).abs() < 1e-12);
            assert!((lhs.y - rhs.y).abs() < 1e-12);
        }
    }
}
