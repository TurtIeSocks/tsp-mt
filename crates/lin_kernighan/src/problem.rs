use crate::{coord::Point2D, error::Result, Error};

/// Symmetric Euclidean TSP problem instance.
///
/// The solver works exclusively in EUC_2D mode. Coordinates are arbitrary
/// `f64` planar values — projection from lat/lng (or any other source) is
/// the caller's responsibility. Integer edge weights are computed lazily
/// via [`distance::euc_2d`](crate::distance::euc_2d) so the input does not
/// need pre-scaling, but for reproducible parity with LKH callers commonly
/// scale by `1000.0` before constructing the problem.
#[derive(Clone, Debug)]
pub struct Problem {
    pub(crate) coords: Vec<Point2D>,
}

impl Problem {
    pub fn new(coords: Vec<Point2D>) -> Result<Self> {
        if coords.len() < 3 {
            return Err(Error::invalid_input(
                "problem requires at least 3 nodes for a tour",
            ));
        }
        if coords.iter().any(|p| !p.is_finite()) {
            return Err(Error::invalid_input("problem contains non-finite coordinates"));
        }
        Ok(Self { coords })
    }

    pub fn n(&self) -> usize {
        self.coords.len()
    }

    pub fn coord(&self, idx: usize) -> Point2D {
        self.coords[idx]
    }

    pub fn coords(&self) -> &[Point2D] {
        &self.coords
    }
}
