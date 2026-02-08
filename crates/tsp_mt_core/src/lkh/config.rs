#![allow(dead_code)]

use std::{
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::Result;

const MAX_TRIALS_MULTIPLIER: usize = 3;
const MIN_MAX_TRIALS: usize = 1_000;
const MAX_MAX_TRIALS: usize = 100_000;
const TIME_LIMIT_DIVISOR: usize = 512;
const MIN_TIME_LIMIT_SECONDS: usize = 2;

const PREPROCESS_RUNS: usize = 1;
const PREPROCESS_TIME_LIMIT_SECONDS: usize = 1;

const SMALL_RUNS: usize = 1;
const SMALL_TRACE_LEVEL: usize = 0;
const SMALL_MAX_CANDIDATES: usize = 32;

const DEFAULT_RUNS: usize = 1;
const DEFAULT_TRACE_LEVEL: usize = 1;
const DEFAULT_MAX_CANDIDATES: usize = 32;
const DEFAULT_BASE_SEED: u64 = 12_345;

/// Yes/No wrapper for LKH parameters expressed as `[ YES | NO ]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum YesNo {
    Yes,
    No,
}

impl YesNo {
    fn as_lkh(self) -> &'static str {
        match self {
            Self::Yes => "YES",
            Self::No => "NO",
        }
    }
}

impl From<bool> for YesNo {
    fn from(value: bool) -> Self {
        if value { Self::Yes } else { Self::No }
    }
}

/// LKH-2.0 `CANDIDATE_SET_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CandidateSetType {
    Alpha,
    Delaunay { pure: bool },
    NearestNeighbor,
    Quadrant,
}

impl CandidateSetType {
    fn as_lkh(self) -> &'static str {
        match self {
            Self::Alpha => "ALPHA",
            Self::Delaunay { pure: false } => "DELAUNAY",
            Self::Delaunay { pure: true } => "DELAUNAY PURE",
            Self::NearestNeighbor => "NEAREST-NEIGHBOR",
            Self::Quadrant => "QUADRANT",
        }
    }
}

/// LKH-2.0 `EXTRA_CANDIDATE_SET_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExtraCandidateSetType {
    NearestNeighbor,
    Quadrant,
}

impl ExtraCandidateSetType {
    fn as_lkh(self) -> &'static str {
        match self {
            Self::NearestNeighbor => "NEAREST-NEIGHBOR",
            Self::Quadrant => "QUADRANT",
        }
    }
}

/// LKH-2.0 `INITIAL_TOUR_ALGORITHM` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InitialTourAlgorithm {
    Boruvka,
    Greedy,
    NearestNeighbor,
    QuickBoruvka,
    Sierpinski,
    Walk,
}

impl InitialTourAlgorithm {
    fn as_lkh(self) -> &'static str {
        match self {
            Self::Boruvka => "BORUVKA",
            Self::Greedy => "GREEDY",
            Self::NearestNeighbor => "NEAREST-NEIGHBOR",
            Self::QuickBoruvka => "QUICK-BORUVKA",
            Self::Sierpinski => "SIERPINSKI",
            Self::Walk => "WALK",
        }
    }
}

/// LKH-2.0 `PATCHING_A` / `PATCHING_C` option modifiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PatchingRuleMode {
    Restricted,
    Extended,
}

impl PatchingRuleMode {
    fn as_lkh(self) -> &'static str {
        match self {
            Self::Restricted => "RESTRICTED",
            Self::Extended => "EXTENDED",
        }
    }
}

/// LKH candidate count with optional `SYMMETRIC` modifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CandidateLimit {
    pub(crate) value: usize,
    pub(crate) symmetric: bool,
}

impl CandidateLimit {
    pub(crate) const fn new(value: usize, symmetric: bool) -> Self {
        Self { value, symmetric }
    }
}

/// LKH patching count with optional `RESTRICTED`/`EXTENDED` mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PatchingRule {
    pub(crate) max_cycles: usize,
    pub(crate) mode: Option<PatchingRuleMode>,
}

impl PatchingRule {
    pub(crate) const fn new(max_cycles: usize, mode: Option<PatchingRuleMode>) -> Self {
        Self { max_cycles, mode }
    }
}

/// LKH-2.0 `SUBPROBLEM_SIZE` partitioning modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SubproblemPartitioning {
    Delaunay,
    Karp,
    KMeans,
    Rohe,
    Sierpinski,
}

impl SubproblemPartitioning {
    fn as_lkh(self) -> &'static str {
        match self {
            Self::Delaunay => "DELAUNAY",
            Self::Karp => "KARP",
            Self::KMeans => "K-MEANS",
            Self::Rohe => "ROHE",
            Self::Sierpinski => "SIERPINSKI",
        }
    }
}

/// Composite value for the `SUBPROBLEM_SIZE` parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SubproblemSpec {
    pub(crate) size: usize,
    pub(crate) partitioning: Option<SubproblemPartitioning>,
    pub(crate) borders: bool,
    pub(crate) compressed: bool,
}

impl SubproblemSpec {
    pub(crate) const fn new(size: usize) -> Self {
        Self {
            size,
            partitioning: None,
            borders: false,
            compressed: false,
        }
    }
}

/// Comment lines accepted by LKH (`COMMENT ...` and `# ...`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ParameterComment {
    CommentKeyword(String),
    HashPrefix(String),
}

/// Full LKH-2.0 parameter-file model.
///
/// The field docs quote or paraphrase the wording from `LKH-2.0_PARAMETERS.pdf`.
#[derive(Clone, Debug)]
pub(crate) struct LkhConfig {
    /// "Specifies the name of the problem file." (mandatory)
    problem_file: PathBuf,

    /// "The number of candidate edges to be associated with each node during the ascent."
    pub(crate) ascent_candidates: Option<usize>,
    /// "The number of backbone trials in each run."
    pub(crate) backbone_trials: Option<usize>,
    /// "Specifies whether a backtracking K-opt move is to be used as the first move."
    pub(crate) backtracking: Option<YesNo>,
    /// "Specifies the name of a file to which the candidate sets are to be written."
    pub(crate) candidate_files: Vec<PathBuf>,
    /// "Specifies the candidate set type."
    pub(crate) candidate_set_type: Option<CandidateSetType>,
    /// "A comment." (`COMMENT <string>`)
    pub(crate) comment_lines: Vec<ParameterComment>,
    /// "Terminates the input data. The entry is optional." (`EOF`)
    pub(crate) emit_eof: bool,
    /// "The maximum alpha-value allowed for any candidate edge is set to EXCESS times ..."
    pub(crate) excess: Option<f64>,
    /// "Number of extra candidate edges ..." optionally followed by `SYMMETRIC`.
    pub(crate) extra_candidates: Option<CandidateLimit>,
    /// "The candidate set type of extra candidate edges."
    pub(crate) extra_candidate_set_type: Option<ExtraCandidateSetType>,
    /// "Specifies whether the Gain23 function is used."
    pub(crate) gain23: Option<YesNo>,
    /// "Specifies whether Lin and Kernighan's gain criterion is used."
    pub(crate) gain_criterion: Option<YesNo>,
    /// "The length of the first period in the ascent."
    pub(crate) initial_period: Option<usize>,
    /// "The initial step size used in the ascent."
    pub(crate) initial_step_size: Option<usize>,
    /// "Specifies the algorithm for obtaining an initial tour."
    pub(crate) initial_tour_algorithm: Option<InitialTourAlgorithm>,
    /// "Specifies the name of a file containing a tour to be used as the initial tour."
    pub(crate) initial_tour_file: Option<PathBuf>,
    /// "Specifies the fraction of the initial tour to be constructed by ... file edges."
    pub(crate) initial_tour_fraction: Option<f64>,
    /// "Specifies the name of a file containing a tour" used to limit search.
    pub(crate) input_tour_file: Option<PathBuf>,
    /// "Specifies the number of times to 'kick' a tour found by Lin-Kernighan."
    pub(crate) kicks: Option<usize>,
    /// "Specifies the value of K for a random K-swap kick."
    pub(crate) kick_type: Option<usize>,
    /// "Specifies the maximum number of candidate edges considered at each search level."
    pub(crate) max_breadth: Option<usize>,
    /// "The maximum number of candidate edges to be associated with each node."
    pub(crate) max_candidates: Option<CandidateLimit>,
    /// "Specifies the maximum number of swaps (flips) allowed ..."
    pub(crate) max_swaps: Option<usize>,
    /// "The maximum number of trials in each run."
    pub(crate) max_trials: Option<usize>,
    /// "Specifies the name of a tour to be merged." (repeatable)
    pub(crate) merge_tour_files: Vec<PathBuf>,
    /// "Specifies the sequential move type to be used in local search."
    pub(crate) move_type: Option<usize>,
    /// "Specifies the nonsequential move type to be used."
    pub(crate) nonsequential_move_type: Option<usize>,
    /// "Specifies the name of a file where the best tour is to be written."
    pub(crate) output_tour_file: Option<PathBuf>,
    /// "Known optimal tour length."
    pub(crate) optimum: Option<f64>,
    /// "The maximum number of disjoint alternating cycles to be used for patching."
    pub(crate) patching_a: Option<PatchingRule>,
    /// "The maximum number of disjoint cycles to be patched ..." (`PATCHING_C`)
    pub(crate) patching_c: Option<PatchingRule>,
    /// "Specifies the name of a file to which penalties (Pi-values) are to be written."
    pub(crate) pi_file: Option<PathBuf>,
    /// "The internal precision in the representation of transformed distances."
    pub(crate) precision: Option<usize>,
    /// "Specifies whether ... search pruning technique is used."
    pub(crate) restricted_search: Option<YesNo>,
    /// "The total number of runs."
    pub(crate) runs: Option<usize>,
    /// "Specifies the initial seed for random number generation."
    pub(crate) seed: Option<u64>,
    /// "Specifies whether a run is stopped, if the tour length becomes equal to OPTIMUM."
    pub(crate) stop_at_optimum: Option<YesNo>,
    /// "Specifies whether the pi-values should be determined by subgradient optimization."
    pub(crate) subgradient: Option<YesNo>,
    /// "The number of nodes in a division of the original problem into subproblems."
    pub(crate) subproblem_size: Option<SubproblemSpec>,
    /// "Specifies the name of a file containing a tour ... used for dividing ... subproblems."
    pub(crate) subproblem_tour_file: Option<PathBuf>,
    /// "Specifies the move type to be used for all moves following the first move ..."
    pub(crate) subsequent_move_type: Option<usize>,
    /// "Specifies whether patching is used for moves following the first move ..."
    pub(crate) subsequent_patching: Option<YesNo>,
    /// "Specifies a time limit in seconds for each run."
    pub(crate) time_limit: Option<f64>,
    /// "Specifies the name of a file to which the best tour is to be written." (`TOUR_FILE`)
    pub(crate) tour_file: Option<PathBuf>,
    /// "Specifies the level of detail of the output given during the solution process."
    pub(crate) trace_level: Option<usize>,
}

impl LkhConfig {
    pub(crate) fn new(problem_file: impl Into<PathBuf>) -> Self {
        Self {
            problem_file: problem_file.into(),
            ascent_candidates: None,
            backbone_trials: None,
            backtracking: None,
            candidate_files: Vec::new(),
            candidate_set_type: None,
            comment_lines: Vec::new(),
            emit_eof: false,
            excess: None,
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
            kicks: None,
            kick_type: None,
            max_breadth: None,
            max_candidates: None,
            max_swaps: None,
            max_trials: None,
            merge_tour_files: Vec::new(),
            move_type: None,
            nonsequential_move_type: None,
            output_tour_file: None,
            optimum: None,
            patching_a: None,
            patching_c: None,
            pi_file: None,
            precision: None,
            restricted_search: None,
            runs: None,
            seed: None,
            stop_at_optimum: None,
            subgradient: None,
            subproblem_size: None,
            subproblem_tour_file: None,
            subsequent_move_type: None,
            subsequent_patching: None,
            time_limit: None,
            tour_file: None,
            trace_level: None,
        }
    }

    /// Runtime defaults used by the parallel solver.
    pub(crate) fn for_parallel_solve(
        n: usize,
        problem_file: impl Into<PathBuf>,
        candidate_file: impl Into<PathBuf>,
        pi_file: impl Into<PathBuf>,
    ) -> Self {
        let mut cfg = Self::new(problem_file);
        cfg.runs = Some(DEFAULT_RUNS);
        cfg.max_trials = Some((n * MAX_TRIALS_MULTIPLIER).clamp(MIN_MAX_TRIALS, MAX_MAX_TRIALS));
        cfg.trace_level = Some(DEFAULT_TRACE_LEVEL);
        cfg.time_limit = Some(((n / TIME_LIMIT_DIVISOR).max(MIN_TIME_LIMIT_SECONDS)) as f64);
        cfg.max_candidates = Some(CandidateLimit::new(DEFAULT_MAX_CANDIDATES, true));
        cfg.seed = Some(DEFAULT_BASE_SEED);
        cfg.candidate_files.push(candidate_file.into());
        cfg.pi_file = Some(pi_file.into());
        cfg
    }

    /// One-run candidate preprocessing used to materialize candidate/PI files.
    pub(crate) fn for_preprocessing(
        n: usize,
        problem_file: impl Into<PathBuf>,
        candidate_file: impl Into<PathBuf>,
        pi_file: impl Into<PathBuf>,
    ) -> Self {
        let mut cfg = Self::new(problem_file);
        cfg.runs = Some(PREPROCESS_RUNS);
        cfg.max_trials = Some(n);
        cfg.trace_level = Some(DEFAULT_TRACE_LEVEL);
        cfg.time_limit = Some(PREPROCESS_TIME_LIMIT_SECONDS as f64);
        cfg.max_candidates = Some(CandidateLimit::new(DEFAULT_MAX_CANDIDATES, true));
        cfg.seed = Some(DEFAULT_BASE_SEED);
        cfg.candidate_files.push(candidate_file.into());
        cfg.pi_file = Some(pi_file.into());
        cfg
    }

    /// Small-problem preset used by chunk-centroid ordering.
    pub(crate) fn for_small_problem(
        problem_file: impl Into<PathBuf>,
        max_trials: usize,
        time_limit_seconds: usize,
    ) -> Self {
        let mut cfg = Self::new(problem_file);
        cfg.runs = Some(SMALL_RUNS);
        cfg.max_trials = Some(max_trials);
        cfg.trace_level = Some(SMALL_TRACE_LEVEL);
        cfg.time_limit = Some(time_limit_seconds as f64);
        cfg.max_candidates = Some(CandidateLimit::new(SMALL_MAX_CANDIDATES, true));
        cfg
    }

    pub(crate) fn base_seed(&self) -> u64 {
        self.seed.unwrap_or(DEFAULT_BASE_SEED)
    }

    pub(crate) fn time_limit_seconds(&self) -> Option<f64> {
        self.time_limit
    }

    pub(crate) fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub(crate) fn with_output_tour_file(mut self, output_tour_file: impl Into<PathBuf>) -> Self {
        self.output_tour_file = Some(output_tour_file.into());
        self
    }

    /// Deterministic seed generation from the configured base seed.
    pub(crate) fn generate_seeds(&self, count: usize) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(self.base_seed());
        (0..count).map(|_| rng.random::<u64>()).collect()
    }

    pub(crate) fn write_to_file(&self, path: &Path) -> Result<()> {
        fs::write(path, self.render())?;
        Ok(())
    }

    pub(crate) fn render(&self) -> String {
        let mut out = String::new();

        writeln!(&mut out, "PROBLEM_FILE = {}", self.problem_file.display()).expect("write");

        push_opt_usize(&mut out, "ASCENT_CANDIDATES", self.ascent_candidates);
        push_opt_usize(&mut out, "BACKBONE_TRIALS", self.backbone_trials);
        push_opt_yes_no(&mut out, "BACKTRACKING", self.backtracking);

        for file in &self.candidate_files {
            writeln!(&mut out, "CANDIDATE_FILE = {}", file.display()).expect("write");
        }

        if let Some(value) = self.candidate_set_type {
            writeln!(&mut out, "CANDIDATE_SET_TYPE = {}", value.as_lkh()).expect("write");
        }

        for comment in &self.comment_lines {
            match comment {
                ParameterComment::CommentKeyword(text) => {
                    writeln!(&mut out, "COMMENT {text}").expect("write");
                }
                ParameterComment::HashPrefix(text) => {
                    writeln!(&mut out, "# {text}").expect("write");
                }
            }
        }

        push_opt_f64(&mut out, "EXCESS", self.excess);

        if let Some(value) = self.extra_candidates {
            push_candidate_limit(&mut out, "EXTRA_CANDIDATES", value);
        }

        if let Some(value) = self.extra_candidate_set_type {
            writeln!(&mut out, "EXTRA_CANDIDATE_SET_TYPE = {}", value.as_lkh()).expect("write");
        }

        push_opt_yes_no(&mut out, "GAIN23", self.gain23);
        push_opt_yes_no(&mut out, "GAIN_CRITERION", self.gain_criterion);
        push_opt_usize(&mut out, "INITIAL_PERIOD", self.initial_period);
        push_opt_usize(&mut out, "INITIAL_STEP_SIZE", self.initial_step_size);

        if let Some(value) = self.initial_tour_algorithm {
            writeln!(&mut out, "INITIAL_TOUR_ALGORITHM = {}", value.as_lkh()).expect("write");
        }

        push_opt_path(
            &mut out,
            "INITIAL_TOUR_FILE",
            self.initial_tour_file.as_deref(),
        );
        push_opt_f64(
            &mut out,
            "INITIAL_TOUR_FRACTION",
            self.initial_tour_fraction,
        );
        push_opt_path(&mut out, "INPUT_TOUR_FILE", self.input_tour_file.as_deref());
        push_opt_usize(&mut out, "KICKS", self.kicks);
        push_opt_usize(&mut out, "KICK_TYPE", self.kick_type);
        push_opt_usize(&mut out, "MAX_BREADTH", self.max_breadth);

        if let Some(value) = self.max_candidates {
            push_candidate_limit(&mut out, "MAX_CANDIDATES", value);
        }

        push_opt_usize(&mut out, "MAX_SWAPS", self.max_swaps);
        push_opt_usize(&mut out, "MAX_TRIALS", self.max_trials);

        for file in &self.merge_tour_files {
            writeln!(&mut out, "MERGE_TOUR_FILE = {}", file.display()).expect("write");
        }

        push_opt_usize(&mut out, "MOVE_TYPE", self.move_type);
        push_opt_usize(
            &mut out,
            "NONSEQUENTIAL_MOVE_TYPE",
            self.nonsequential_move_type,
        );
        push_opt_path(
            &mut out,
            "OUTPUT_TOUR_FILE",
            self.output_tour_file.as_deref(),
        );
        push_opt_f64(&mut out, "OPTIMUM", self.optimum);

        if let Some(value) = self.patching_a {
            push_patching_rule(&mut out, "PATCHING_A", value);
        }
        if let Some(value) = self.patching_c {
            push_patching_rule(&mut out, "PATCHING_C", value);
        }

        push_opt_path(&mut out, "PI_FILE", self.pi_file.as_deref());
        push_opt_usize(&mut out, "PRECISION", self.precision);
        push_opt_yes_no(&mut out, "RESTRICTED_SEARCH", self.restricted_search);
        push_opt_usize(&mut out, "RUNS", self.runs);

        if let Some(value) = self.seed {
            writeln!(&mut out, "SEED = {value}").expect("write");
        }

        push_opt_yes_no(&mut out, "STOP_AT_OPTIMUM", self.stop_at_optimum);
        push_opt_yes_no(&mut out, "SUBGRADIENT", self.subgradient);

        if let Some(value) = self.subproblem_size {
            let mut line = format!("SUBPROBLEM_SIZE = {}", value.size);
            if let Some(partitioning) = value.partitioning {
                line.push(' ');
                line.push_str(partitioning.as_lkh());
            }
            if value.borders {
                line.push_str(" BORDERS");
            }
            if value.compressed {
                line.push_str(" COMPRESSED");
            }
            writeln!(&mut out, "{line}").expect("write");
        }

        push_opt_path(
            &mut out,
            "SUBPROBLEM_TOUR_FILE",
            self.subproblem_tour_file.as_deref(),
        );
        push_opt_usize(&mut out, "SUBSEQUENT_MOVE_TYPE", self.subsequent_move_type);
        push_opt_yes_no(&mut out, "SUBSEQUENT_PATCHING", self.subsequent_patching);
        push_opt_f64(&mut out, "TIME_LIMIT", self.time_limit);
        push_opt_path(&mut out, "TOUR_FILE", self.tour_file.as_deref());
        push_opt_usize(&mut out, "TRACE_LEVEL", self.trace_level);

        if self.emit_eof {
            out.push_str("EOF\n");
        }

        out
    }
}

fn push_candidate_limit(out: &mut String, key: &str, value: CandidateLimit) {
    if value.symmetric {
        writeln!(out, "{key} = {} SYMMETRIC", value.value).expect("write");
    } else {
        writeln!(out, "{key} = {}", value.value).expect("write");
    }
}

fn push_patching_rule(out: &mut String, key: &str, value: PatchingRule) {
    let mut line = format!("{key} = {}", value.max_cycles);
    if let Some(mode) = value.mode {
        line.push(' ');
        line.push_str(mode.as_lkh());
    }
    writeln!(out, "{line}").expect("write");
}

fn push_opt_path(out: &mut String, key: &str, value: Option<&Path>) {
    if let Some(path) = value {
        writeln!(out, "{key} = {}", path.display()).expect("write");
    }
}

fn push_opt_usize(out: &mut String, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        writeln!(out, "{key} = {value}").expect("write");
    }
}

fn push_opt_yes_no(out: &mut String, key: &str, value: Option<YesNo>) {
    if let Some(value) = value {
        writeln!(out, "{key} = {}", value.as_lkh()).expect("write");
    }
}

fn push_opt_f64(out: &mut String, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        writeln!(out, "{key} = {value}").expect("write");
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        CandidateLimit, CandidateSetType, LkhConfig, MAX_MAX_TRIALS, MAX_TRIALS_MULTIPLIER,
        MIN_MAX_TRIALS, MIN_TIME_LIMIT_SECONDS, ParameterComment,
    };

    #[test]
    fn parallel_config_clamps_trials_and_time_limit() {
        let small = LkhConfig::for_parallel_solve(1, "problem.tsp", "problem.cand", "problem.pi");
        let large =
            LkhConfig::for_parallel_solve(1_000_000, "problem.tsp", "problem.cand", "problem.pi");

        let small_text = small.render();
        let large_text = large.render();

        assert!(small_text.contains(&format!("MAX_TRIALS = {MIN_MAX_TRIALS}")));
        assert!(small_text.contains(&format!("TIME_LIMIT = {MIN_TIME_LIMIT_SECONDS}")));
        assert!(large_text.contains(&format!("MAX_TRIALS = {MAX_MAX_TRIALS}")));
        assert!(!large_text.contains(&format!(
            "MAX_TRIALS = {}",
            1_000_000 * MAX_TRIALS_MULTIPLIER
        )));
    }

    #[test]
    fn preprocessing_uses_single_run_and_fixed_time_limit() {
        let cfg = LkhConfig::for_preprocessing(123, "problem.tsp", "problem.cand", "problem.pi");
        let text = cfg.render();
        assert!(text.contains("RUNS = 1"));
        assert!(text.contains("MAX_TRIALS = 123"));
        assert!(text.contains("TIME_LIMIT = 1"));
    }

    #[test]
    fn generate_seeds_is_deterministic_for_default_seed() {
        let cfg = LkhConfig::for_parallel_solve(10, "problem.tsp", "problem.cand", "problem.pi");
        let a = cfg.generate_seeds(4);
        let b = cfg.generate_seeds(4);
        assert_eq!(a, b);
    }

    #[test]
    fn render_emits_comprehensive_parameter_lines() {
        let mut cfg = LkhConfig::new("problem.tsp");
        cfg.max_candidates = Some(CandidateLimit::new(32, true));
        cfg.candidate_set_type = Some(CandidateSetType::Delaunay { pure: true });
        cfg.candidate_files.push(PathBuf::from("problem.cand"));
        cfg.comment_lines
            .push(ParameterComment::CommentKeyword("sample".to_string()));
        cfg.comment_lines
            .push(ParameterComment::HashPrefix("sample2".to_string()));
        cfg.output_tour_file = Some(PathBuf::from("run.tour"));
        cfg.emit_eof = true;

        let text = cfg.render();
        assert!(text.contains("PROBLEM_FILE = problem.tsp"));
        assert!(text.contains("MAX_CANDIDATES = 32 SYMMETRIC"));
        assert!(text.contains("CANDIDATE_SET_TYPE = DELAUNAY PURE"));
        assert!(text.contains("CANDIDATE_FILE = problem.cand"));
        assert!(text.contains("COMMENT sample"));
        assert!(text.contains("# sample2"));
        assert!(text.contains("OUTPUT_TOUR_FILE = run.tour"));
        assert!(text.ends_with("EOF\n"));
    }
}
