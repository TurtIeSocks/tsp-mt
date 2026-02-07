pub(crate) const MIN_CYCLE_POINTS: usize = 3;

pub(crate) const PAR_EXTENSION: &str = ".par";
pub(crate) const TOUR_EXTENSION: &str = ".tour";
pub(crate) const TSP_EXTENSION: &str = ".tsp";
pub(crate) const CANDIDATE_EXTENSION: &str = ".cand";
pub(crate) const PI_EXTENSION: &str = ".pi";

pub(crate) struct File {
    name: &'static str,
}

impl File {
    const fn new(name: &'static str) -> Self {
        Self { name }
    }
    fn file_name(&self, ext: &'static str) -> String {
        format!("{}{ext}", self.name)
    }
    fn idx_file_name(&self, idx: usize, ext: &'static str) -> String {
        format!("{}_{idx}{ext}", self.name)
    }

    pub fn par(&self) -> String {
        self.file_name(PAR_EXTENSION)
    }
    pub fn par_idx(&self, idx: usize) -> String {
        self.idx_file_name(idx, PAR_EXTENSION)
    }
    pub fn tour(&self) -> String {
        self.file_name(TOUR_EXTENSION)
    }
    pub fn tour_idx(&self, idx: usize) -> String {
        self.idx_file_name(idx, TOUR_EXTENSION)
    }
    pub fn candidate(&self) -> String {
        self.file_name(CANDIDATE_EXTENSION)
    }
    pub fn pi(&self) -> String {
        self.file_name(PI_EXTENSION)
    }
    pub fn tsp(&self) -> String {
        self.file_name(TSP_EXTENSION)
    }
    pub const fn name(&self) -> &'static str {
        self.name
    }
}

pub(crate) const PROBLEM_FILE: File = File::new("problem");
pub(crate) const PREP_CANDIDATES_FILE: File = File::new("prep_candidates");
pub(crate) const RUN_FILE: File = File::new("run");
pub(crate) const CENTROIDS_FILE: File = File::new("centroids");
