use std::path::PathBuf;

use crate::lkh::Point;

/// Runtime input for LKH solver.
#[derive(Clone, Debug)]
pub struct SolverInput<'a> {
    pub(crate) lkh_exe: &'a PathBuf,
    pub(crate) work_dir: &'a PathBuf,
    pub(crate) points: &'a [Point],
}

impl<'a> SolverInput<'a> {
    pub fn new(lkh_exe: &'a PathBuf, work_dir: &'a PathBuf, points: &'a [Point]) -> Self {
        Self {
            lkh_exe,
            work_dir,
            points,
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.points.len()
    }

    pub(crate) fn get_point(&self, idx: usize) -> Point {
        self.points[idx]
    }
}
