use lkh::{
    LkhResult,
    parameters::{CandidateLimit, LkhParameters},
    problem::{EdgeWeightType, NodeCoord, TsplibProblem, TsplibProblemType},
    solver::LkhSolver,
};

use crate::{LKHNode, SolverInput, SolverOptions};

pub fn lkh_single(input: SolverInput, options: SolverOptions) -> LkhResult<Vec<LKHNode>> {
    let problem_file_path = options.work_dir.join("problem.tsp");

    let tour = LkhSolver::new(
        TsplibProblem::new(TsplibProblemType::Tsp)
            .with_node_coord_section(
                input
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(idx, n)| NodeCoord::twod(idx + 1, n.y, n.x))
                    .collect::<Vec<_>>(),
            )
            .with_dimension(input.n())
            .with_edge_weight_type(EdgeWeightType::Geo),
        LkhParameters::new(&problem_file_path)
            .with_max_candidates(CandidateLimit::new(32, true))
            .with_max_trials(1000)
            .with_runs(1)
            .with_seed(12345)
            .with_time_limit(5.0)
            .with_trace_level(2),
    )?
    .run()?;

    Ok(tour
        .zero_based_tour()?
        .into_iter()
        .map(|idx| input.get_point(idx))
        .collect())
}

pub fn lkh_multi_parallel(input: SolverInput, options: SolverOptions) -> LkhResult<Vec<LKHNode>> {
    todo!()
}

pub fn lkh_multi(input: SolverInput, options: SolverOptions) -> LkhResult<Vec<LKHNode>> {
    todo!()
}
