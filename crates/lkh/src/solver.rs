use std::{
    fs,
    path::{Path, PathBuf},
};

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

impl LkhSolver {
    pub fn new(problem: TsplibProblem, mut params: LkhParameters) -> LkhResult<Self> {
        let workdir = derive_workdir(&params.problem_file);

        if params.output_tour_file.is_none() {
            params.output_tour_file = Some(workdir.join(DEFAULT_TOUR_FILE));
        }

        let solver = Self {
            workdir,
            problem,
            params,
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
        let old_workdir = self.workdir.clone();
        self.workdir = workdir.into();
        rebase_parameter_paths(&mut self.params, &old_workdir, &self.workdir);
    }

    fn _run(&self, process: LkhProcess) -> LkhResult<TsplibTour> {
        let param_file = self.workdir.join(DEFAULT_PARAMS_FILE);
        let Some(ref tour_file) = self.params.output_tour_file else {
            return Err(LkhError::other("missing tour file"));
        };

        fs::create_dir_all(&self.workdir)?;

        self.problem.write_to_file(&self.params.problem_file)?;
        self.params.write_to_file(&param_file)?;

        process
            .with_current_dir(self.workdir.clone())
            .run(param_file)?;
        let tour = TsplibTour::from_file(tour_file)?;
        Ok(tour)
    }
}

fn derive_workdir(problem_file: &Path) -> PathBuf {
    match problem_file.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn rebase_parameter_paths(params: &mut LkhParameters, old_workdir: &Path, new_workdir: &Path) {
    params.problem_file = rebase_path(&params.problem_file, old_workdir, new_workdir);
    params.candidate_files = params
        .candidate_files
        .iter()
        .map(|path| rebase_path(path, old_workdir, new_workdir))
        .collect();
    params.initial_tour_file = params
        .initial_tour_file
        .as_ref()
        .map(|path| rebase_path(path, old_workdir, new_workdir));
    params.input_tour_file = params
        .input_tour_file
        .as_ref()
        .map(|path| rebase_path(path, old_workdir, new_workdir));
    params.merge_tour_files = params
        .merge_tour_files
        .iter()
        .map(|path| rebase_path(path, old_workdir, new_workdir))
        .collect();
    params.output_tour_file = params
        .output_tour_file
        .as_ref()
        .map(|path| rebase_path(path, old_workdir, new_workdir));
    params.pi_file = params
        .pi_file
        .as_ref()
        .map(|path| rebase_path(path, old_workdir, new_workdir));
    params.subproblem_tour_file = params
        .subproblem_tour_file
        .as_ref()
        .map(|path| rebase_path(path, old_workdir, new_workdir));
    params.tour_file = params
        .tour_file
        .as_ref()
        .map(|path| rebase_path(path, old_workdir, new_workdir));
}

fn rebase_path(path: &Path, old_workdir: &Path, new_workdir: &Path) -> PathBuf {
    if let Ok(relative_to_old) = path.strip_prefix(old_workdir) {
        return new_workdir.join(relative_to_old);
    }

    if path.is_absolute() {
        return path.to_path_buf();
    }

    new_workdir.join(path)
}

#[cfg(test)]
mod tests {
    use super::{LkhSolver, derive_workdir};
    use crate::{
        parameters::LkhParameters,
        problem::{TsplibProblem, TsplibProblemType},
    };

    use std::path::PathBuf;

    #[test]
    fn derive_workdir_defaults_to_current_directory_marker() {
        assert_eq!(
            derive_workdir(PathBuf::from("problem.tsp").as_path()),
            PathBuf::from(".")
        );
        assert_eq!(
            derive_workdir(PathBuf::from("a/problem.tsp").as_path()),
            PathBuf::from("a")
        );
    }

    #[test]
    fn set_workdir_rebases_all_parameter_paths() {
        let mut params = LkhParameters::new("old/problem.tsp");
        params.candidate_files.push(PathBuf::from("cand1.txt"));
        params.candidate_files.push(PathBuf::from("old/cand2.txt"));
        params.initial_tour_file = Some(PathBuf::from("seed.tour"));
        params.input_tour_file = Some(PathBuf::from("old/input.tour"));
        params.merge_tour_files = vec![PathBuf::from("m1.tour"), PathBuf::from("old/m2.tour")];
        params.output_tour_file = Some(PathBuf::from("run.tour"));
        params.pi_file = Some(PathBuf::from("old/problem.pi"));
        params.subproblem_tour_file = Some(PathBuf::from("sub.tour"));
        params.tour_file = Some(PathBuf::from("old/final.tour"));

        let mut solver = LkhSolver::new(TsplibProblem::new(TsplibProblemType::Tsp), params)
            .expect("solver should build");
        solver.set_workdir("new");

        assert_eq!(solver.workdir, PathBuf::from("new"));
        assert_eq!(solver.params.problem_file, PathBuf::from("new/problem.tsp"));
        assert_eq!(
            solver.params.candidate_files,
            vec![
                PathBuf::from("new/cand1.txt"),
                PathBuf::from("new/cand2.txt")
            ]
        );
        assert_eq!(
            solver.params.initial_tour_file,
            Some(PathBuf::from("new/seed.tour"))
        );
        assert_eq!(
            solver.params.input_tour_file,
            Some(PathBuf::from("new/input.tour"))
        );
        assert_eq!(
            solver.params.merge_tour_files,
            vec![PathBuf::from("new/m1.tour"), PathBuf::from("new/m2.tour")]
        );
        assert_eq!(
            solver.params.output_tour_file,
            Some(PathBuf::from("new/run.tour"))
        );
        assert_eq!(solver.params.pi_file, Some(PathBuf::from("new/problem.pi")));
        assert_eq!(
            solver.params.subproblem_tour_file,
            Some(PathBuf::from("new/sub.tour"))
        );
        assert_eq!(
            solver.params.tour_file,
            Some(PathBuf::from("new/final.tour"))
        );
    }

    #[test]
    fn set_workdir_keeps_external_absolute_paths() {
        let external = std::env::temp_dir().join("external.tour");
        let mut params = LkhParameters::new("old/problem.tsp");
        params.output_tour_file = Some(external.clone());

        let mut solver = LkhSolver::new(TsplibProblem::new(TsplibProblemType::Tsp), params)
            .expect("solver should build");
        solver.set_workdir("new");

        assert_eq!(solver.params.output_tour_file, Some(external));
    }
}
