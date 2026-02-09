use std::{
    fmt::{Display, Formatter},
    fs,
    path::{Path, PathBuf},
};

use crate::{LkhResult, spec_writer::SpecWriter};
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

/// LKH-2.0 `CANDIDATE_SET_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum CandidateSetType {
    Alpha,
    #[lkh(value = "DELAUNAY", pat = "{ pure: false }")]
    #[lkh(value = "DELAUNAY PURE", pat = "{ pure: true }")]
    Delaunay {
        pure: bool,
    },
    NearestNeighbor,
    Quadrant,
}

/// LKH-2.0 `EXTRA_CANDIDATE_SET_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum ExtraCandidateSetType {
    NearestNeighbor,
    Quadrant,
}

/// LKH-2.0 `INITIAL_TOUR_ALGORITHM` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum InitialTourAlgorithm {
    Boruvka,
    Greedy,
    NearestNeighbor,
    QuickBoruvka,
    Sierpinski,
    Walk,
}

/// LKH-2.0 `PATCHING_A` / `PATCHING_C` option modifiers.
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

/// LKH-2.0 `SUBPROBLEM_SIZE` partitioning modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum SubproblemPartitioning {
    Delaunay,
    Karp,
    KMeans,
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

/// Full LKH-2.0 parameter-file model.
///
/// The field docs quote or paraphrase the wording from `LKH-2.0_PARAMETERS.pdf`.
#[derive(Clone, Debug, WithMethods)]
pub struct LkhParameters {
    /// "Specifies the name of the problem file." (mandatory)
    problem_file: PathBuf,

    /// "The number of candidate edges to be associated with each node during the ascent."
    pub ascent_candidates: Option<usize>,
    /// "The number of backbone trials in each run."
    pub backbone_trials: Option<usize>,
    /// "Specifies whether a backtracking K-opt move is to be used as the first move."
    pub backtracking: Option<YesNo>,
    /// "Specifies the name of a file to which the candidate sets are to be written."
    pub candidate_files: Vec<PathBuf>,
    /// "Specifies the candidate set type."
    pub candidate_set_type: Option<CandidateSetType>,
    /// "A comment." (`COMMENT <string>`)
    pub comment_lines: Vec<ParameterComment>,
    /// "Terminates the input data. The entry is optional." (`EOF`)
    pub emit_eof: bool,
    /// "The maximum alpha-value allowed for any candidate edge is set to EXCESS times ..."
    pub excess: Option<f64>,
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
    /// "Specifies the number of times to 'kick' a tour found by Lin-Kernighan."
    pub kicks: Option<usize>,
    /// "Specifies the value of K for a random K-swap kick."
    pub kick_type: Option<usize>,
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
    /// "The internal precision in the representation of transformed distances."
    pub precision: Option<usize>,
    /// "Specifies whether ... search pruning technique is used."
    pub restricted_search: Option<YesNo>,
    /// "The total number of runs."
    pub runs: Option<usize>,
    /// "Specifies the initial seed for random number generation."
    pub seed: Option<u64>,
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
    /// "Specifies whether patching is used for moves following the first move ..."
    pub subsequent_patching: Option<YesNo>,
    /// "Specifies a time limit in seconds for each run."
    pub time_limit: Option<f64>,
    /// "Specifies the name of a file to which the best tour is to be written." (`TOUR_FILE`)
    pub tour_file: Option<PathBuf>,
    /// "Specifies the level of detail of the output given during the solution process."
    pub trace_level: Option<usize>,
}

impl Display for LkhParameters {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = SpecWriter::new(f);

        // PROBLEM_FILE must always be first.
        writer.kv_eq("PROBLEM_FILE", self.problem_file.display())?;

        // Keep the remaining keys alphabetical for stable, testable output.
        writer.opt_kv_eq("ASCENT_CANDIDATES", self.ascent_candidates)?;
        writer.opt_kv_eq("BACKBONE_TRIALS", self.backbone_trials)?;
        writer.opt_kv_eq("BACKTRACKING", self.backtracking)?;
        writer.path_list_eq("CANDIDATE_FILE", &self.candidate_files)?;
        writer.opt_kv_eq("CANDIDATE_SET_TYPE", self.candidate_set_type)?;
        writer.lines("", &self.comment_lines)?;
        writer.opt_kv_eq("EXCESS", self.excess)?;
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
        writer.opt_kv_eq("KICKS", self.kicks)?;
        writer.opt_kv_eq("KICK_TYPE", self.kick_type)?;
        writer.opt_kv_eq("MAX_BREADTH", self.max_breadth)?;
        writer.opt_kv_eq("MAX_CANDIDATES", self.max_candidates)?;
        writer.opt_kv_eq("MAX_SWAPS", self.max_swaps)?;
        writer.opt_kv_eq("MAX_TRIALS", self.max_trials)?;
        writer.path_list_eq("MERGE_TOUR_FILE", &self.merge_tour_files)?;
        writer.opt_kv_eq("MOVE_TYPE", self.move_type)?;
        writer.opt_kv_eq("NONSEQUENTIAL_MOVE_TYPE", self.nonsequential_move_type)?;
        writer.opt_kv_eq("OPTIMUM", self.optimum)?;
        writer.opt_path_eq("OUTPUT_TOUR_FILE", self.output_tour_file.as_ref())?;
        writer.opt_kv_eq("PATCHING_A", self.patching_a)?;
        writer.opt_kv_eq("PATCHING_C", self.patching_c)?;
        writer.opt_path_eq("PI_FILE", self.pi_file.as_ref())?;
        writer.opt_kv_eq("PRECISION", self.precision)?;
        writer.opt_kv_eq("RESTRICTED_SEARCH", self.restricted_search)?;
        writer.opt_kv_eq("RUNS", self.runs)?;
        writer.opt_kv_eq("SEED", self.seed)?;
        writer.opt_kv_eq("STOP_AT_OPTIMUM", self.stop_at_optimum)?;
        writer.opt_kv_eq("SUBGRADIENT", self.subgradient)?;
        writer.opt_kv_eq("SUBPROBLEM_SIZE", self.subproblem_size)?;
        writer.opt_path_eq("SUBPROBLEM_TOUR_FILE", self.subproblem_tour_file.as_ref())?;
        writer.opt_kv_eq("SUBSEQUENT_MOVE_TYPE", self.subsequent_move_type)?;
        writer.opt_kv_eq("SUBSEQUENT_PATCHING", self.subsequent_patching)?;
        writer.opt_kv_eq("TIME_LIMIT", self.time_limit)?;
        writer.opt_path_eq("TOUR_FILE", self.tour_file.as_ref())?;
        writer.opt_kv_eq("TRACE_LEVEL", self.trace_level)?;

        // EOF is optional and must be last when enabled.
        if self.emit_eof {
            writer.line("EOF")?;
        }

        Ok(())
    }
}

impl LkhParameters {
    pub fn new(problem_file: impl Into<PathBuf>) -> Self {
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

    pub fn write_to_file(&self, path: &Path) -> LkhResult<()> {
        fs::write(path, self.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        CandidateLimit, CandidateSetType, ExtraCandidateSetType, InitialTourAlgorithm,
        LkhParameters, ParameterComment, PatchingRule, PatchingRuleMode, SubproblemPartitioning,
        SubproblemSpec, YesNo,
    };
    use lkh_derive::WithMethods;

    #[derive(Clone, Debug, WithMethods)]
    struct SkipFixture {
        keep: Option<u64>,
        #[with(skip)]
        skip: Option<u64>,
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
}
