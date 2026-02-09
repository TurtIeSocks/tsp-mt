use std::{fs, path::PathBuf};

use crate::{
    LkhError, LkhResult, parameters::LkhParameters, problem::TsplibProblem, process::LkhProcess,
    tour::TsplibTour,
};

#[derive(Clone)]
pub struct LkhSolver {
    workdir: PathBuf,
    problem: TsplibProblem,
    params: LkhParameters,
}

const DEFAULT_TOUR_FILE: &str = "problem.tour";
const DEFAULT_PARAMS_FILE: &str = "problem.par";
const DEFAULT_PROBLEM_FILE: &str = "problem.tsp";

impl LkhSolver {
    pub fn new(problem: TsplibProblem, params: LkhParameters) -> LkhResult<Self> {
        let mut workdir = params.problem_file.clone();
        workdir.pop();

        let tour_file = if let Some(ref tf) = params.output_tour_file {
            tf.clone()
        } else {
            workdir.join(DEFAULT_TOUR_FILE)
        };

        let solver: LkhSolver = Self {
            workdir,
            problem,
            params: params.try_with_output_tour_file(tour_file)?,
        };

        Ok(solver)
    }

    #[cfg(feature = "embedded-lkh")]
    pub fn run(&self) -> LkhResult<TsplibTour> {
        self._run(LkhProcess::try_default()?)
    }

    pub fn run_with_exe(&self, exe_path: impl Into<PathBuf>) -> LkhResult<TsplibTour> {
        self._run(LkhProcess::new(exe_path))
    }

    pub fn set_workdir(&mut self, workdir: impl Into<PathBuf>) {
        self.workdir = workdir.into();
        let problem_file_name = self.params.problem_file.file_name();
        let problem_file = if let Some(name) = problem_file_name {
            self.workdir.join(name)
        } else {
            self.workdir.join(DEFAULT_PROBLEM_FILE)
        };
        self.params.problem_file = problem_file;
    }

    fn _run(&self, process: LkhProcess) -> LkhResult<TsplibTour> {
        let param_file = self.workdir.join(DEFAULT_PARAMS_FILE);
        let Some(ref tour_file) = self.params.output_tour_file else {
            return Err(LkhError::other("missing tour file"));
        };

        fs::create_dir_all(&self.workdir)?;

        self.problem.write_to_file(&self.params.problem_file)?;
        self.params.write_to_file(&param_file)?;

        process.run(param_file)?;
        let tour = TsplibTour::from_file(tour_file)?;
        Ok(tour)
    }
}
