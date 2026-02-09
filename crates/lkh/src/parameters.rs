//! LKH parameter-file (`.par`) model and writer.
//!
//! For a full key/value format reference for this crate, see:
//! `crates/lkh/docs/PAR_SPEC.md`.

use std::{
    fmt::{Display, Formatter},
    fs,
    path::PathBuf,
};

use crate::{LkhError, LkhResult, spec_writer::SpecWriter, with_methods_error};
use lkh_derive::{LkhDisplay, WithMethods};

/// Yes/No wrapper for LKH parameters expressed as `[ YES | NO ]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum YesNo {
    Yes,
    No,
}

impl From<bool> for YesNo {
    fn from(value: bool) -> Self {
        if value { Self::Yes } else { Self::No }
    }
}

/// LKH-3 `CANDIDATE_SET_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum CandidateSetType {
    Alpha,
    #[lkh(value = "DELAUNAY", pat = "{ pure: false }")]
    #[lkh(value = "DELAUNAY PURE", pat = "{ pure: true }")]
    Delaunay {
        pure: bool,
    },
    NearestNeighbor,
    Popmusic,
    Quadrant,
}

/// LKH-3 `EXTRA_CANDIDATE_SET_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum ExtraCandidateSetType {
    NearestNeighbor,
    Popmusic,
    Quadrant,
}

/// LKH-3 `INITIAL_TOUR_ALGORITHM` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum InitialTourAlgorithm {
    Boruvka,
    Ctsp,
    Cvrp,
    Gctsp,
    Greedy,
    Moore,
    Mtsp,
    NearestNeighbor,
    Pctsp,
    QuickBoruvka,
    Sierpinski,
    Sop,
    Tspdl,
    Walk,
}

/// LKH-3 `MTSP_OBJECTIVE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
#[lkh(separator = "_")]
pub enum MtspObjective {
    Minmax,
    MinmaxSize,
    Minsum,
}

/// LKH-3 `RECOMBINATION` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum Recombination {
    Ipt,
    Gpx2,
    Clarist,
}

/// LKH-3 `PATCHING_A` / `PATCHING_C` option modifiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum PatchingRuleMode {
    Restricted,
    Extended,
}

/// LKH candidate count with optional `SYMMETRIC` modifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateLimit {
    pub value: usize,
    pub symmetric: bool,
}

impl CandidateLimit {
    pub const fn new(value: usize, symmetric: bool) -> Self {
        Self { value, symmetric }
    }
}

impl Display for CandidateLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)?;
        if self.symmetric {
            write!(f, " SYMMETRIC")
        } else {
            Ok(())
        }
    }
}

/// LKH patching count with optional `RESTRICTED`/`EXTENDED` mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PatchingRule {
    pub max_cycles: usize,
    pub mode: Option<PatchingRuleMode>,
}

impl PatchingRule {
    pub const fn new(max_cycles: usize, mode: Option<PatchingRuleMode>) -> Self {
        Self { max_cycles, mode }
    }
}

impl Display for PatchingRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.max_cycles)?;
        if let Some(mode) = self.mode {
            write!(f, " {mode}")?;
        }
        Ok(())
    }
}

/// LKH-3 `BWTSP` parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BwtspSpec {
    pub b: usize,
    pub q: usize,
    pub l: Option<usize>,
}

impl BwtspSpec {
    pub const fn new(b: usize, q: usize, l: Option<usize>) -> Self {
        Self { b, q, l }
    }
}

impl Display for BwtspSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.b, self.q)?;
        if let Some(l) = self.l {
            write!(f, " {l}")?;
        }
        Ok(())
    }
}

/// LKH-3 `SUBPROBLEM_SIZE` partitioning modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum SubproblemPartitioning {
    Delaunay,
    Karp,
    KCenter,
    KMeans,
    Moore,
    Rohe,
    Sierpinski,
}

/// Composite value for the `SUBPROBLEM_SIZE` parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubproblemSpec {
    pub size: usize,
    pub partitioning: Option<SubproblemPartitioning>,
    pub borders: bool,
    pub compressed: bool,
}

impl SubproblemSpec {
    pub const fn new(size: usize) -> Self {
        Self {
            size,
            partitioning: None,
            borders: false,
            compressed: false,
        }
    }
}

impl Display for SubproblemSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.size)?;
        if let Some(partitioning) = self.partitioning {
            write!(f, " {partitioning}")?;
        }
        if self.borders {
            write!(f, " BORDERS")?;
        }
        if self.compressed {
            write!(f, " COMPRESSED")?;
        }

        Ok(())
    }
}

/// Comment lines accepted by LKH (`COMMENT ...` and `# ...`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParameterComment {
    CommentKeyword(String),
    HashPrefix(String),
}

impl Display for ParameterComment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterComment::CommentKeyword(text) => {
                write!(f, "COMMENT {text}")
            }
            ParameterComment::HashPrefix(text) => {
                write!(f, "# {text}")
            }
        }
    }
}

/// Full LKH-3 parameter-file model.
///
/// The field docs quote or paraphrase wording from the LKH-3 parameter guide.
#[derive(Clone, Debug, WithMethods)]
#[with(error = LkhError)]
pub struct LkhParameters {
    /// "Specifies the name of the problem file." (mandatory)
    pub(crate) problem_file: PathBuf,

    /// "The number of candidate edges to be associated with each node during the ascent."
    pub ascent_candidates: Option<usize>,
    /// "The number of backbone trials in each run."
    pub backbone_trials: Option<usize>,
    /// "Specifies whether a backtracking K-opt move is to be used as the first move."
    pub backtracking: Option<YesNo>,
    /// "Specifies the three parameters (B, Q, L) to a BWTSP instance."
    pub bwtsp: Option<BwtspSpec>,
    /// "Specifies the name of a file to which the candidate sets are to be written."
    pub candidate_files: Vec<PathBuf>,
    /// "Specifies the candidate set type."
    pub candidate_set_type: Option<CandidateSetType>,
    /// "A comment." (`COMMENT <string>`)
    pub comment_lines: Vec<ParameterComment>,
    /// "Specifies the depot node."
    pub depot: Option<usize>,
    /// "The maximum route length for distance-constrained variants."
    pub distance: Option<f64>,
    /// "Specifies file(s) of candidate edges in Concorde format." (repeatable)
    pub edge_files: Vec<PathBuf>,
    /// "Terminates the input data. The entry is optional." (`EOF`)
    pub emit_eof: bool,
    /// "The maximum alpha-value allowed for any candidate edge is set to EXCESS times ..."
    pub excess: Option<f64>,
    /// "Specifies the number of external salesmen for an OCMTSP problem."
    pub external_salesmen: Option<usize>,
    /// "Number of extra candidate edges ..." optionally followed by `SYMMETRIC`.
    pub extra_candidates: Option<CandidateLimit>,
    /// "The candidate set type of extra candidate edges."
    pub extra_candidate_set_type: Option<ExtraCandidateSetType>,
    /// "Specifies whether the Gain23 function is used."
    pub gain23: Option<YesNo>,
    /// "Specifies whether Lin and Kernighan's gain criterion is used."
    pub gain_criterion: Option<YesNo>,
    /// "The length of the first period in the ascent."
    pub initial_period: Option<usize>,
    /// "The initial step size used in the ascent."
    pub initial_step_size: Option<usize>,
    /// "Specifies the algorithm for obtaining an initial tour."
    pub initial_tour_algorithm: Option<InitialTourAlgorithm>,
    /// "Specifies the name of a file containing a tour to be used as the initial tour."
    pub initial_tour_file: Option<PathBuf>,
    /// "Specifies the fraction of the initial tour to be constructed by ... file edges."
    pub initial_tour_fraction: Option<f64>,
    /// "Specifies the name of a file containing a tour" used to limit search.
    pub input_tour_file: Option<PathBuf>,
    /// "The k-value used for solving a k-TSP instance."
    pub k: Option<usize>,
    /// "Specifies the number of times to 'kick' a tour found by Lin-Kernighan."
    pub kicks: Option<usize>,
    /// "Specifies the value of K for a random K-swap kick."
    pub kick_type: Option<usize>,
    /// "Specifies if makespan optimization is to be used for a TSPTW instance."
    pub makespan: Option<YesNo>,
    /// "Specifies the maximum number of candidate edges considered at each search level."
    pub max_breadth: Option<usize>,
    /// "The maximum number of candidate edges to be associated with each node."
    pub max_candidates: Option<CandidateLimit>,
    /// "Specifies the maximum number of swaps (flips) allowed ..."
    pub max_swaps: Option<usize>,
    /// "The maximum number of trials in each run."
    pub max_trials: Option<usize>,
    /// "Specifies the name of a tour to be merged." (repeatable)
    pub merge_tour_files: Vec<PathBuf>,
    /// "Specifies the sequential move type to be used in local search."
    pub move_type: Option<usize>,
    /// Optional `SPECIAL` suffix for `MOVE_TYPE`.
    pub move_type_special: bool,
    /// "Specifies the maximum number of cities each salesman may visit in MTSP/VRP."
    pub mtsp_max_size: Option<usize>,
    /// "Specifies the minimum number of cities each salesman must visit in MTSP/VRP."
    pub mtsp_min_size: Option<i64>,
    /// "Specifies the objective function type for a multiple traveling salesman problem."
    pub mtsp_objective: Option<MtspObjective>,
    /// "Specifies the file where MTSP/VRP solutions are to be written."
    pub mtsp_solution_file: Option<PathBuf>,
    /// "Specifies the nonsequential move type to be used."
    pub nonsequential_move_type: Option<usize>,
    /// "Specifies the name of a file where the best tour is to be written."
    pub output_tour_file: Option<PathBuf>,
    /// "Known optimal tour length."
    pub optimum: Option<f64>,
    /// "The maximum number of disjoint alternating cycles to be used for patching."
    pub patching_a: Option<PatchingRule>,
    /// "The maximum number of disjoint cycles to be patched ..." (`PATCHING_C`)
    pub patching_c: Option<PatchingRule>,
    /// "Specifies the name of a file to which penalties (Pi-values) are to be written."
    pub pi_file: Option<PathBuf>,
    /// "Specifies whether the best POPMUSIC tour is to be used as initial tour."
    pub popmusic_initial_tour: Option<YesNo>,
    /// "Maximum number of nearest neighbors used as POPMUSIC candidates."
    pub popmusic_max_neighbors: Option<usize>,
    /// "POPMUSIC sample size."
    pub popmusic_sample_size: Option<usize>,
    /// "POPMUSIC number of solutions to generate."
    pub popmusic_solutions: Option<usize>,
    /// "POPMUSIC iterated 3-opt trials."
    pub popmusic_trials: Option<usize>,
    /// "Maximum population size for the genetic algorithm."
    pub population_size: Option<usize>,
    /// "The internal precision in the representation of transformed distances."
    pub precision: Option<usize>,
    /// "The probability (percent) for PTSP."
    pub probability: Option<usize>,
    /// "Specifies recombination mode."
    pub recombination: Option<Recombination>,
    /// "Specifies whether ... search pruning technique is used."
    pub restricted_search: Option<YesNo>,
    /// "The total number of runs."
    pub runs: Option<usize>,
    /// "Specifies the number of salesmen/vehicles."
    pub salesmen: Option<usize>,
    /// "Scale factor. Distances are multiplied by this factor."
    pub scale: Option<usize>,
    /// "Specifies the initial seed for random number generation."
    pub seed: Option<u64>,
    /// "Specifies the file where MTSP/VRP solutions are written in SINTEF format."
    pub sintef_solution_file: Option<PathBuf>,
    /// Equivalent to LKH's `SPECIAL` macro keyword in parameter files.
    pub special: bool,
    /// "Specifies whether a run is stopped, if the tour length becomes equal to OPTIMUM."
    pub stop_at_optimum: Option<YesNo>,
    /// "Specifies whether the pi-values should be determined by subgradient optimization."
    pub subgradient: Option<YesNo>,
    /// "The number of nodes in a division of the original problem into subproblems."
    pub subproblem_size: Option<SubproblemSpec>,
    /// "Specifies the name of a file containing a tour ... used for dividing ... subproblems."
    pub subproblem_tour_file: Option<PathBuf>,
    /// "Specifies the move type to be used for all moves following the first move ..."
    pub subsequent_move_type: Option<usize>,
    /// Optional `SPECIAL` suffix for `SUBSEQUENT_MOVE_TYPE`.
    pub subsequent_move_type_special: bool,
    /// "Specifies whether patching is used for moves following the first move ..."
    pub subsequent_patching: Option<YesNo>,
    /// "Specifies a time limit in seconds for each run."
    pub time_limit: Option<f64>,
    /// "Specifies a total time limit in seconds."
    pub total_time_limit: Option<f64>,
    /// "Specifies the name of a file to which the best tour is to be written." (`TOUR_FILE`)
    pub tour_file: Option<PathBuf>,
    /// "Specifies the level of detail of the output given during the solution process."
    pub trace_level: Option<usize>,
    /// Alias of `SALESMEN` accepted by LKH.
    pub vehicles: Option<usize>,
}

with_methods_error!(LkhParametersWithMethodsError);

impl Display for LkhParameters {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = SpecWriter::new(f);

        // PROBLEM_FILE must always be first.
        writer.kv_eq("PROBLEM_FILE", self.problem_file.display())?;

        // Keep the remaining keys alphabetical for stable, testable output.
        writer.opt_kv_eq("ASCENT_CANDIDATES", self.ascent_candidates)?;
        writer.opt_kv_eq("BACKBONE_TRIALS", self.backbone_trials)?;
        writer.opt_kv_eq("BACKTRACKING", self.backtracking)?;
        writer.opt_kv_eq("BWTSP", self.bwtsp)?;
        writer.path_list_eq("CANDIDATE_FILE", &self.candidate_files)?;
        writer.opt_kv_eq("CANDIDATE_SET_TYPE", self.candidate_set_type)?;
        writer.lines("", &self.comment_lines)?;
        writer.opt_kv_eq("DEPOT", self.depot)?;
        writer.opt_kv_eq("DISTANCE", self.distance)?;
        writer.path_list_eq("EDGE_FILE", &self.edge_files)?;
        writer.opt_kv_eq("EXCESS", self.excess)?;
        writer.opt_kv_eq("EXTERNAL_SALESMEN", self.external_salesmen)?;
        writer.opt_kv_eq("EXTRA_CANDIDATES", self.extra_candidates)?;
        writer.opt_kv_eq("EXTRA_CANDIDATE_SET_TYPE", self.extra_candidate_set_type)?;
        writer.opt_kv_eq("GAIN23", self.gain23)?;
        writer.opt_kv_eq("GAIN_CRITERION", self.gain_criterion)?;
        writer.opt_kv_eq("INITIAL_PERIOD", self.initial_period)?;
        writer.opt_kv_eq("INITIAL_STEP_SIZE", self.initial_step_size)?;
        writer.opt_kv_eq("INITIAL_TOUR_ALGORITHM", self.initial_tour_algorithm)?;
        writer.opt_path_eq("INITIAL_TOUR_FILE", self.initial_tour_file.as_ref())?;
        writer.opt_kv_eq("INITIAL_TOUR_FRACTION", self.initial_tour_fraction)?;
        writer.opt_path_eq("INPUT_TOUR_FILE", self.input_tour_file.as_ref())?;
        writer.opt_kv_eq("K", self.k)?;
        writer.opt_kv_eq("KICKS", self.kicks)?;
        writer.opt_kv_eq("KICK_TYPE", self.kick_type)?;
        writer.opt_kv_eq("MAKESPAN", self.makespan)?;
        writer.opt_kv_eq("MAX_BREADTH", self.max_breadth)?;
        writer.opt_kv_eq("MAX_CANDIDATES", self.max_candidates)?;
        writer.opt_kv_eq("MAX_SWAPS", self.max_swaps)?;
        writer.opt_kv_eq("MAX_TRIALS", self.max_trials)?;
        writer.path_list_eq("MERGE_TOUR_FILE", &self.merge_tour_files)?;

        if let Some(move_type) = self.move_type {
            let value = if self.move_type_special {
                format!("{move_type} SPECIAL")
            } else {
                move_type.to_string()
            };
            writer.kv_eq("MOVE_TYPE", value)?;
        }

        writer.opt_kv_eq("MTSP_MAX_SIZE", self.mtsp_max_size)?;
        writer.opt_kv_eq("MTSP_MIN_SIZE", self.mtsp_min_size)?;
        writer.opt_kv_eq("MTSP_OBJECTIVE", self.mtsp_objective)?;
        writer.opt_path_eq("MTSP_SOLUTION_FILE", self.mtsp_solution_file.as_ref())?;
        writer.opt_kv_eq("NONSEQUENTIAL_MOVE_TYPE", self.nonsequential_move_type)?;
        writer.opt_kv_eq("OPTIMUM", self.optimum)?;
        writer.opt_path_eq("OUTPUT_TOUR_FILE", self.output_tour_file.as_ref())?;
        writer.opt_kv_eq("PATCHING_A", self.patching_a)?;
        writer.opt_kv_eq("PATCHING_C", self.patching_c)?;
        writer.opt_path_eq("PI_FILE", self.pi_file.as_ref())?;
        writer.opt_kv_eq("POPMUSIC_INITIAL_TOUR", self.popmusic_initial_tour)?;
        writer.opt_kv_eq("POPMUSIC_MAX_NEIGHBORS", self.popmusic_max_neighbors)?;
        writer.opt_kv_eq("POPMUSIC_SAMPLE_SIZE", self.popmusic_sample_size)?;
        writer.opt_kv_eq("POPMUSIC_SOLUTIONS", self.popmusic_solutions)?;
        writer.opt_kv_eq("POPMUSIC_TRIALS", self.popmusic_trials)?;
        writer.opt_kv_eq("POPULATION_SIZE", self.population_size)?;
        writer.opt_kv_eq("PRECISION", self.precision)?;
        writer.opt_kv_eq("PROBABILITY", self.probability)?;
        writer.opt_kv_eq("RECOMBINATION", self.recombination)?;
        writer.opt_kv_eq("RESTRICTED_SEARCH", self.restricted_search)?;
        writer.opt_kv_eq("RUNS", self.runs)?;
        writer.opt_kv_eq("SALESMEN", self.salesmen)?;
        writer.opt_kv_eq("SCALE", self.scale)?;
        writer.opt_kv_eq("SEED", self.seed)?;
        writer.opt_path_eq("SINTEF_SOLUTION_FILE", self.sintef_solution_file.as_ref())?;

        if self.special {
            writer.line("SPECIAL")?;
        }

        writer.opt_kv_eq("STOP_AT_OPTIMUM", self.stop_at_optimum)?;
        writer.opt_kv_eq("SUBGRADIENT", self.subgradient)?;
        writer.opt_kv_eq("SUBPROBLEM_SIZE", self.subproblem_size)?;
        writer.opt_path_eq("SUBPROBLEM_TOUR_FILE", self.subproblem_tour_file.as_ref())?;

        if let Some(subsequent_move_type) = self.subsequent_move_type {
            let value = if self.subsequent_move_type_special {
                format!("{subsequent_move_type} SPECIAL")
            } else {
                subsequent_move_type.to_string()
            };
            writer.kv_eq("SUBSEQUENT_MOVE_TYPE", value)?;
        }

        writer.opt_kv_eq("SUBSEQUENT_PATCHING", self.subsequent_patching)?;
        writer.opt_kv_eq("TIME_LIMIT", self.time_limit)?;
        writer.opt_kv_eq("TOTAL_TIME_LIMIT", self.total_time_limit)?;
        writer.opt_path_eq("TOUR_FILE", self.tour_file.as_ref())?;
        writer.opt_kv_eq("TRACE_LEVEL", self.trace_level)?;
        writer.opt_kv_eq("VEHICLES", self.vehicles)?;

        // EOF is optional and must be last when enabled.
        if self.emit_eof {
            writer.line("EOF")?;
        }

        Ok(())
    }
}

impl LkhParameters {
    /// Creates a parameter model with the required `PROBLEM_FILE` set.
    pub fn new(problem_file: impl Into<PathBuf>) -> Self {
        Self {
            problem_file: problem_file.into(),
            ascent_candidates: None,
            backbone_trials: None,
            backtracking: None,
            bwtsp: None,
            candidate_files: Vec::new(),
            candidate_set_type: None,
            comment_lines: Vec::new(),
            depot: None,
            distance: None,
            edge_files: Vec::new(),
            emit_eof: true,
            excess: None,
            external_salesmen: None,
            extra_candidates: None,
            extra_candidate_set_type: None,
            gain23: None,
            gain_criterion: None,
            initial_period: None,
            initial_step_size: None,
            initial_tour_algorithm: None,
            initial_tour_file: None,
            initial_tour_fraction: None,
            input_tour_file: None,
            k: None,
            kicks: None,
            kick_type: None,
            makespan: None,
            max_breadth: None,
            max_candidates: None,
            max_swaps: None,
            max_trials: None,
            merge_tour_files: Vec::new(),
            move_type: None,
            move_type_special: false,
            mtsp_max_size: None,
            mtsp_min_size: None,
            mtsp_objective: None,
            mtsp_solution_file: None,
            nonsequential_move_type: None,
            output_tour_file: None,
            optimum: None,
            patching_a: None,
            patching_c: None,
            pi_file: None,
            popmusic_initial_tour: None,
            popmusic_max_neighbors: None,
            popmusic_sample_size: None,
            popmusic_solutions: None,
            popmusic_trials: None,
            population_size: None,
            precision: None,
            probability: None,
            recombination: None,
            restricted_search: None,
            runs: None,
            salesmen: None,
            scale: None,
            seed: None,
            sintef_solution_file: None,
            special: false,
            stop_at_optimum: None,
            subgradient: None,
            subproblem_size: None,
            subproblem_tour_file: None,
            subsequent_move_type: None,
            subsequent_move_type_special: false,
            subsequent_patching: None,
            time_limit: None,
            total_time_limit: None,
            tour_file: None,
            trace_level: None,
            vehicles: None,
        }
    }

    /// Serializes and writes this parameter file to disk.
    pub fn write_to_file(&self, file_path: impl Into<PathBuf>) -> LkhResult<()> {
        fs::write(file_path.into(), self.to_string()).map_err(LkhError::Io)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        BwtspSpec, CandidateLimit, CandidateSetType, ExtraCandidateSetType, InitialTourAlgorithm,
        LkhParameters, MtspObjective, ParameterComment, PatchingRule, PatchingRuleMode,
        Recombination, SubproblemPartitioning, SubproblemSpec, YesNo,
    };
    use lkh_derive::WithMethods;

    #[derive(Clone, Debug, WithMethods)]
    struct SkipFixture {
        keep: Option<u64>,
        #[with(skip)]
        skip: Option<u64>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum FixtureError {
        AlreadySet(&'static str),
    }

    #[derive(Clone, Debug, Eq, PartialEq, WithMethods)]
    #[with(error = FixtureError)]
    struct CustomErrorFixture {
        value: Option<u64>,
    }

    impl From<CustomErrorFixtureWithMethodsError> for FixtureError {
        fn from(value: CustomErrorFixtureWithMethodsError) -> Self {
            match value {
                CustomErrorFixtureWithMethodsError(field) => Self::AlreadySet(field),
            }
        }
    }

    impl SkipFixture {
        fn new() -> Self {
            Self {
                keep: None,
                skip: None,
            }
        }

        fn with_skip(mut self, skip: u64) -> Self {
            self.skip = Some(skip);
            self
        }
    }

    impl CustomErrorFixture {
        fn new() -> Self {
            Self { value: None }
        }
    }

    #[test]
    fn display_emits_comprehensive_parameter_lines() {
        let mut cfg = LkhParameters::new("problem.tsp");
        cfg.max_candidates = Some(CandidateLimit::new(32, true));
        cfg.candidate_set_type = Some(CandidateSetType::Delaunay { pure: true });
        cfg.candidate_files.push(PathBuf::from("problem.cand"));
        cfg.comment_lines
            .push(ParameterComment::CommentKeyword("sample".to_string()));
        cfg.comment_lines
            .push(ParameterComment::HashPrefix("sample2".to_string()));
        cfg.output_tour_file = Some(PathBuf::from("run.tour"));
        cfg.emit_eof = true;

        let text = cfg.to_string();
        assert!(text.contains("PROBLEM_FILE = problem.tsp"));
        assert!(text.contains("MAX_CANDIDATES = 32 SYMMETRIC"));
        assert!(text.contains("CANDIDATE_SET_TYPE = DELAUNAY PURE"));
        assert!(text.contains("CANDIDATE_FILE = problem.cand"));
        assert!(text.contains("COMMENT sample"));
        assert!(text.contains("# sample2"));
        assert!(text.contains("OUTPUT_TOUR_FILE = run.tour"));
        assert!(text.ends_with("EOF\n"));
    }

    #[test]
    fn display_orders_problem_file_first_then_alphabetical() {
        let mut cfg = LkhParameters::new("problem.tsp");
        cfg.ascent_candidates = Some(1);
        cfg.backbone_trials = Some(2);
        cfg.backtracking = Some(YesNo::Yes);
        cfg.candidate_files.push(PathBuf::from("problem.cand"));
        cfg.candidate_set_type = Some(CandidateSetType::Alpha);
        cfg.comment_lines
            .push(ParameterComment::CommentKeyword("sample".to_string()));
        cfg.excess = Some(1.5);
        cfg.extra_candidate_set_type = Some(ExtraCandidateSetType::Quadrant);
        cfg.extra_candidates = Some(CandidateLimit::new(3, false));
        cfg.gain23 = Some(YesNo::No);
        cfg.gain_criterion = Some(YesNo::Yes);
        cfg.initial_period = Some(4);
        cfg.initial_step_size = Some(5);
        cfg.initial_tour_algorithm = Some(InitialTourAlgorithm::Greedy);
        cfg.initial_tour_file = Some(PathBuf::from("initial.tour"));
        cfg.input_tour_file = Some(PathBuf::from("input.tour"));
        cfg.initial_tour_fraction = Some(0.5);
        cfg.kicks = Some(6);
        cfg.kick_type = Some(7);
        cfg.max_breadth = Some(8);
        cfg.max_candidates = Some(CandidateLimit::new(9, true));
        cfg.max_swaps = Some(10);
        cfg.max_trials = Some(11);
        cfg.merge_tour_files.push(PathBuf::from("merge.tour"));
        cfg.move_type = Some(12);
        cfg.nonsequential_move_type = Some(13);
        cfg.optimum = Some(14.0);
        cfg.output_tour_file = Some(PathBuf::from("output.tour"));
        cfg.patching_a = Some(PatchingRule::new(15, Some(PatchingRuleMode::Extended)));
        cfg.patching_c = Some(PatchingRule::new(16, None));
        cfg.pi_file = Some(PathBuf::from("problem.pi"));
        cfg.precision = Some(17);
        cfg.restricted_search = Some(YesNo::No);
        cfg.runs = Some(18);
        cfg.seed = Some(19);
        cfg.stop_at_optimum = Some(YesNo::Yes);
        cfg.subgradient = Some(YesNo::No);
        cfg.subproblem_size = Some(SubproblemSpec {
            size: 20,
            partitioning: Some(SubproblemPartitioning::KMeans),
            borders: true,
            compressed: true,
        });
        cfg.subproblem_tour_file = Some(PathBuf::from("subproblem.tour"));
        cfg.subsequent_move_type = Some(21);
        cfg.subsequent_patching = Some(YesNo::Yes);
        cfg.time_limit = Some(22.0);
        cfg.tour_file = Some(PathBuf::from("tour.out"));
        cfg.trace_level = Some(23);

        let text = cfg.to_string();
        let lines: Vec<&str> = text.lines().collect();
        let keys: Vec<&str> = lines
            .iter()
            .filter_map(|line| {
                if let Some((key, _)) = line.split_once(" = ") {
                    Some(key)
                } else if line.starts_with("COMMENT ") {
                    Some("COMMENT")
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(keys.first(), Some(&"PROBLEM_FILE"));
        let mut sorted = keys[1..].to_vec();
        sorted.sort_unstable();
        assert_eq!(&keys[1..], sorted.as_slice());
    }

    #[test]
    fn display_emits_lkh3_parameter_extensions() {
        let mut cfg = LkhParameters::new("problem.tsp");
        cfg.bwtsp = Some(BwtspSpec::new(3, 4, Some(5)));
        cfg.depot = Some(7);
        cfg.distance = Some(123.0);
        cfg.edge_files.push(PathBuf::from("edges.txt"));
        cfg.external_salesmen = Some(2);
        cfg.k = Some(9);
        cfg.makespan = Some(YesNo::Yes);
        cfg.move_type = Some(5);
        cfg.move_type_special = true;
        cfg.mtsp_max_size = Some(100);
        cfg.mtsp_min_size = Some(-1);
        cfg.mtsp_objective = Some(MtspObjective::MinmaxSize);
        cfg.mtsp_solution_file = Some(PathBuf::from("mtsp.sol"));
        cfg.popmusic_initial_tour = Some(YesNo::No);
        cfg.popmusic_max_neighbors = Some(8);
        cfg.popmusic_sample_size = Some(20);
        cfg.popmusic_solutions = Some(40);
        cfg.popmusic_trials = Some(2);
        cfg.population_size = Some(10);
        cfg.probability = Some(75);
        cfg.recombination = Some(Recombination::Clarist);
        cfg.salesmen = Some(3);
        cfg.scale = Some(1000);
        cfg.sintef_solution_file = Some(PathBuf::from("sintef.sol"));
        cfg.special = true;
        cfg.subsequent_move_type = Some(3);
        cfg.subsequent_move_type_special = true;
        cfg.total_time_limit = Some(300.0);
        cfg.vehicles = Some(4);

        let text = cfg.to_string();
        assert!(text.contains("BWTSP = 3 4 5"));
        assert!(text.contains("DEPOT = 7"));
        assert!(text.contains("DISTANCE = 123"));
        assert!(text.contains("EDGE_FILE = edges.txt"));
        assert!(text.contains("EXTERNAL_SALESMEN = 2"));
        assert!(text.contains("K = 9"));
        assert!(text.contains("MAKESPAN = YES"));
        assert!(text.contains("MOVE_TYPE = 5 SPECIAL"));
        assert!(text.contains("MTSP_MAX_SIZE = 100"));
        assert!(text.contains("MTSP_MIN_SIZE = -1"));
        assert!(text.contains("MTSP_OBJECTIVE = MINMAX_SIZE"));
        assert!(text.contains("MTSP_SOLUTION_FILE = mtsp.sol"));
        assert!(text.contains("POPMUSIC_INITIAL_TOUR = NO"));
        assert!(text.contains("POPMUSIC_MAX_NEIGHBORS = 8"));
        assert!(text.contains("POPMUSIC_SAMPLE_SIZE = 20"));
        assert!(text.contains("POPMUSIC_SOLUTIONS = 40"));
        assert!(text.contains("POPMUSIC_TRIALS = 2"));
        assert!(text.contains("POPULATION_SIZE = 10"));
        assert!(text.contains("PROBABILITY = 75"));
        assert!(text.contains("RECOMBINATION = CLARIST"));
        assert!(text.contains("SALESMEN = 3"));
        assert!(text.contains("SCALE = 1000"));
        assert!(text.contains("SINTEF_SOLUTION_FILE = sintef.sol"));
        assert!(text.contains("\nSPECIAL\n"));
        assert!(text.contains("SUBSEQUENT_MOVE_TYPE = 3 SPECIAL"));
        assert!(text.contains("TOTAL_TIME_LIMIT = 300"));
        assert!(text.contains("VEHICLES = 4"));
    }

    #[test]
    fn with_methods_set_fields() {
        let cfg = LkhParameters::new("problem.tsp")
            .with_seed(7_u8)
            .with_output_tour_file("out.tour")
            .with_emit_eof(true)
            .with_problem_file("other.tsp");

        assert_eq!(cfg.seed, Some(7));
        assert_eq!(cfg.output_tour_file, Some(PathBuf::from("out.tour")));
        assert!(cfg.emit_eof);
        assert_eq!(cfg.problem_file, PathBuf::from("other.tsp"));
    }

    #[test]
    fn with_methods_skip_attr_excludes_method() {
        let fixture = SkipFixture::new().with_keep(11_u8).with_skip(22);
        assert_eq!(fixture.keep, Some(11));
        assert_eq!(fixture.skip, Some(22));
    }

    #[test]
    fn try_with_methods_set_option_once() {
        let cfg = LkhParameters::new("problem.tsp")
            .try_with_seed(7_u8)
            .expect("first set should succeed");

        assert_eq!(cfg.seed, Some(7));
    }

    #[test]
    fn try_with_methods_fail_when_option_already_set() {
        let err = LkhParameters::new("problem.tsp")
            .with_seed(7_u8)
            .try_with_seed(8_u8)
            .expect_err("second set should fail");

        match err {
            super::LkhError::AlreadyAssigned(message) => {
                assert_eq!(message, "field already set: seed")
            }
            other => panic!("expected AlreadyAssigned, got {other:?}"),
        }
    }

    #[test]
    fn try_with_methods_support_custom_error_type() {
        let err = CustomErrorFixture::new()
            .with_value(1_u8)
            .try_with_value(2_u8)
            .expect_err("second set should fail");

        assert_eq!(err, FixtureError::AlreadySet("value"));
    }
}
