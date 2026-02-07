use std::path::PathBuf;

use crate::LKHNode;

/// Runtime input for LKH solver.
#[derive(Clone, Debug)]
pub struct SolverInput<'a> {
    pub(crate) lkh_exe: &'a PathBuf,
    pub(crate) work_dir: &'a PathBuf,
    pub(crate) points: &'a [LKHNode],
}

impl<'a> SolverInput<'a> {
    pub fn new(lkh_exe: &'a PathBuf, work_dir: &'a PathBuf, points: &'a [LKHNode]) -> Self {
        Self {
            lkh_exe,
            work_dir,
            points,
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.points.len()
    }

    pub(crate) fn get_point(&self, idx: usize) -> LKHNode {
        self.points[idx]
    }
}
