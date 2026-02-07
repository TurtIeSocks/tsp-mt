use geo::Coord;
use map_3d::{self, Ellipsoid};

use crate::utils::Point;

type Precision = f64;
type Geocentric = (Precision, Precision, Precision);
type Topocentric = (Precision, Precision);

pub struct Plane {
    center: Geocentric,
    x: Geocentric,
    y: Geocentric,
    z: Geocentric,
    radius: Precision,
    adjusted_radius: Precision,
    points: Vec<Geocentric>,
}

impl Default for Plane {
    fn default() -> Self {
        Plane {
            center: (0., 0., 0.),
            x: (0., 0., 0.),
            y: (0., 0., 0.),
            z: (0., 0., 0.),
            radius: 0.,
            adjusted_radius: 0.,
            points: vec![],
        }
    }
}

impl Plane {
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
        let t = 1. - Ellipsoid::default().parameters().2;
        (
            (p.2 / (t * t * (p.0 * p.0 + p.1 * p.1).sqrt())).atan(),
            p.1.atan2(p.0),
        )
    }

    fn compute_plane_center(&self) -> Topocentric {
        let mut dir = (0., 0., 0.);
        for (x, y, z) in &self.points {
            dir.0 += x;
            dir.1 += y;
            dir.2 += z;
        }
        self.radial_project(dir)
    }

    pub fn new(input: &Vec<Point>) -> Plane {
        let mut plane = Plane {
            points: input
                .iter()
                .map(|p| {
                    map_3d::geodetic2ecef(
                        p.lat.to_radians(),
                        p.lng.to_radians(),
                        0.,
                        Ellipsoid::default(),
                    )
                })
                .collect(),
            ..Default::default()
        };
        let (plane_center_lat, plane_center_lon) = plane.compute_plane_center();

        // log::debug!(
        //     "Center: {:?}, {:?}",
        //     plane_center_lat.to_degrees(),
        //     plane_center_lon.to_degrees()
        // );

        plane.center =
            map_3d::geodetic2ecef(plane_center_lat, plane_center_lon, 0., Ellipsoid::default());
        plane.z = (
            plane_center_lat.cos() * plane_center_lon.cos(),
            plane_center_lat.cos() * plane_center_lon.sin(),
            plane_center_lat.sin(),
        );
        plane.y = plane.normalize((-plane.center.1, plane.center.0, 0.));
        plane.x = plane.cross_product(plane.z, plane.y);

        plane
    }

    pub fn radius(mut self, radius: Precision) -> Self {
        self.radius = radius;
        let earth_minor = Ellipsoid::default().parameters().1;
        self.adjusted_radius = 0.5 * earth_minor * (2. * self.radius / earth_minor).sin();
        self
    }

    pub fn project(&self) -> Vec<Coord> {
        let global_scale = self.dot_product(self.center, self.z) / self.adjusted_radius;
        let offset_x = self.dot_product(self.center, self.x) / self.adjusted_radius;
        self.points
            .iter()
            .map(|p| {
                let scale = global_scale / self.dot_product(*p, self.z);
                Coord {
                    x: self.dot_product(*p, self.x) * scale - offset_x,
                    y: self.dot_product(*p, self.y) * scale,
                }
            })
            .collect()
    }

    pub fn reverse(&self, input: Vec<Point>) -> Vec<Point> {
        let mut min = 1. / 0.;
        let mut sum = 0.;
        let mut ouput = vec![];
        for p in input.iter() {
            let x = self.center.0 + (self.x.0 * p.lat + self.y.0 * p.lng) * self.adjusted_radius;
            let y = self.center.1 + (self.x.1 * p.lat + self.y.1 * p.lng) * self.adjusted_radius;
            let z = self.center.2 + (self.x.2 * p.lat + self.y.2 * p.lng) * self.adjusted_radius;
            let (lat, lon) = self.radial_project((x, y, z));
            let s = self.dot_product((x, y, z), self.z) / self.euclidean_norm2((x, y, z)).sqrt();

            ouput.push(Point {
                lat: lat.to_degrees(),
                lng: lon.to_degrees(),
            });
            if s < min {
                min = s;
            }
            sum += s;
        }
        eprintln!(
            "Worst scaling: {} (larger/closer to 1 = better; larger area to cover is worse)",
            min
        );
        eprintln!("Average scaling: {}", sum / input.len() as Precision);
        eprintln!("Disc scaling: {}", self.adjusted_radius / self.radius);

        // ouput.sort_by(|a, b| {
        //     geohash::encode(Coord { x: a.lng, y: a.lat }, 9)
        //         .unwrap()
        //         .cmp(&geohash::encode(Coord { x: b.lng, y: b.lat }, 9).unwrap())
        // });

        ouput
    }
}
