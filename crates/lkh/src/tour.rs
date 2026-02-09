//! TSPLIB tour-file parser/writer.
//!
//! For the exact parsing and formatting behavior, see:
//! `crates/lkh/docs/TSPLIB_SPEC.md`.

use std::{
    fmt::{Display, Formatter},
    fs,
    path::{Path, PathBuf},
};

use crate::{LkhError, LkhResult, spec_writer::SpecWriter, with_methods_error};
use lkh_derive::{LkhDisplay, WithMethods};

const TOUR_SECTION_HEADER: &str = "TOUR_SECTION";
const TOUR_END_MARKER: &str = "-1";
const EOF_MARKER: &str = "EOF";
const MIN_VALID_TSPLIB_NODE_ID: isize = 1;
const TSPLIB_NODE_ID_OFFSET: usize = 1;

/// TSPLIB `.tour` `TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum TsplibTourType {
    Tour,
}

/// TSPLIB `.tour` file model.
#[derive(Clone, Debug, PartialEq, WithMethods)]
#[with(error = LkhError)]
pub struct TsplibTour {
    pub name: Option<String>,
    pub comment_lines: Vec<String>,
    pub tour_type: Option<TsplibTourType>,
    pub dimension: Option<usize>,
    /// Optional known optimum (`OPTIMUM`) when present in tour files.
    pub optimum: Option<f64>,
    /// Node identifiers exactly as stored in TSPLIB (1-based).
    pub tour_section: Vec<usize>,
    pub emit_eof: bool,
}

with_methods_error!(TsplibTourWithMethodsError);

impl TsplibTour {
    /// Creates an empty tour model.
    pub fn new() -> Self {
        Self {
            name: None,
            comment_lines: Vec::new(),
            tour_type: None,
            dimension: None,
            optimum: None,
            tour_section: Vec::new(),
            emit_eof: true,
        }
    }

    /// Reads and parses a TSPLIB/LKH tour file from disk.
    pub fn from_file(file_path: impl Into<PathBuf>) -> LkhResult<Self> {
        Self::new().parse_from_file(file_path)
    }

    /// Parses a TSPLIB/LKH tour from text content.
    pub fn from_text(text: String) -> LkhResult<Self> {
        Self::parse(text)
    }

    /// Replaces this tour by parsing a file from disk.
    pub fn read_file(&mut self, file_path: impl Into<PathBuf>) -> LkhResult<()> {
        let text = Self::read(file_path)?;
        *self = Self::parse(text)?;
        Ok(())
    }

    /// Parses a tour file and returns only the zero-based node order.
    pub fn parse_tsplib_tour(path: &Path) -> LkhResult<Vec<usize>> {
        Self::new().parse_from_file(path)?.zero_based_tour()
    }

    /// Returns the tour converted from TSPLIB's 1-based ids to 0-based ids.
    ///
    /// Parsing is intentionally permissive: unknown headers are ignored and
    /// non-positive node ids in `TOUR_SECTION` (except the `-1` terminator) are skipped.
    pub fn zero_based_tour(&self) -> LkhResult<Vec<usize>> {
        let mut zero_based = Vec::with_capacity(self.tour_section.len());
        for &id in &self.tour_section {
            if id < TSPLIB_NODE_ID_OFFSET {
                return Err(LkhError::invalid_data(format!(
                    "Bad node id {id}; TSPLIB ids must be >= {TSPLIB_NODE_ID_OFFSET}"
                )));
            }
            zero_based.push(id - TSPLIB_NODE_ID_OFFSET);
        }

        Ok(zero_based)
    }

    /// Serializes and writes this tour to disk.
    pub fn write_to_file(&self, file_path: impl Into<PathBuf>) -> LkhResult<()> {
        fs::write(file_path.into(), self.to_string()).map_err(LkhError::Io)
    }

    fn read(file_path: impl Into<PathBuf>) -> LkhResult<String> {
        fs::read_to_string(file_path.into()).map_err(LkhError::Io)
    }

    fn parse_from_file(self, file_path: impl Into<PathBuf>) -> LkhResult<Self> {
        let text = Self::read(file_path)?;
        Self::parse(text)
    }

    fn parse(text: String) -> LkhResult<Self> {
        let mut tour = Self::new();
        tour.emit_eof = false;
        let mut in_tour_section = false;
        let mut tour_terminated = false;

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            if line.eq_ignore_ascii_case(EOF_MARKER) {
                tour.emit_eof = true;
                break;
            }

            if !in_tour_section {
                if line.eq_ignore_ascii_case(TOUR_SECTION_HEADER) {
                    in_tour_section = true;
                    continue;
                }

                if let Some((key, value)) = line
                    .split_once(':')
                    .or_else(|| line.split_once('='))
                    .map(|(key, value)| (key.trim().to_ascii_uppercase(), value.trim()))
                {
                    match key.as_str() {
                        "NAME" => {
                            tour.name = Some(value.to_string());
                        }
                        "COMMENT" => {
                            tour.comment_lines.push(value.to_string());
                        }
                        "TYPE" => {
                            if value.eq_ignore_ascii_case("TOUR") {
                                tour.tour_type = Some(TsplibTourType::Tour);
                            } else {
                                return Err(LkhError::invalid_data(format!(
                                    "Unsupported tour TYPE '{value}'"
                                )));
                            }
                        }
                        "DIMENSION" => {
                            let parsed = value.parse::<usize>().map_err(|e| {
                                LkhError::invalid_data(format!(
                                    "Bad DIMENSION value '{value}': {e}"
                                ))
                            })?;
                            tour.dimension = Some(parsed);
                        }
                        "OPTIMUM" => {
                            let parsed = value.parse::<f64>().map_err(|e| {
                                LkhError::invalid_data(format!("Bad OPTIMUM value '{value}': {e}"))
                            })?;
                            tour.optimum = Some(parsed);
                        }
                        _ => {}
                    }
                }

                continue;
            }

            for token in line.split_whitespace() {
                if token == TOUR_END_MARKER {
                    tour_terminated = true;
                    break;
                }
                if token.eq_ignore_ascii_case(EOF_MARKER) {
                    tour.emit_eof = true;
                    tour_terminated = true;
                    break;
                }

                let id: isize = token.parse().map_err(|e| {
                    LkhError::invalid_data(format!("Bad tour token '{token}': {e}"))
                })?;

                if id < MIN_VALID_TSPLIB_NODE_ID {
                    continue;
                }
                tour.tour_section.push(id as usize);
            }

            if tour_terminated {
                break;
            }
        }

        if !in_tour_section {
            return Err(LkhError::invalid_data("Missing TOUR_SECTION"));
        }

        if let Some(dimension) = tour.dimension
            && dimension != tour.tour_section.len()
        {
            return Err(LkhError::invalid_data(format!(
                "DIMENSION is {dimension}, but TOUR_SECTION has {} nodes",
                tour.tour_section.len()
            )));
        }

        Ok(tour)
    }
}

impl Default for TsplibTour {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for TsplibTour {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = SpecWriter::new(f);

        writer.opt_kv_colon("NAME", self.name.as_deref())?;
        writer.opt_kv_colon("TYPE", self.tour_type)?;

        for comment in &self.comment_lines {
            writer.kv_colon("COMMENT", comment)?;
        }

        writer.opt_kv_colon("DIMENSION", self.dimension)?;
        writer.opt_kv_colon("OPTIMUM", self.optimum)?;

        if !self.tour_section.is_empty() {
            writer.lines(TOUR_SECTION_HEADER, &self.tour_section)?;
            writer.line(TOUR_END_MARKER)?;
        }

        if self.emit_eof {
            writer.line(EOF_MARKER)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{TsplibTour, TsplibTourType};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("lkh-tests-{name}-{nanos}"))
    }

    #[test]
    fn parse_tsplib_tour_reads_tour_section_and_converts_to_zero_based() {
        let dir = unique_temp_dir("parse-ok");
        fs::create_dir_all(&dir).expect("create temp dir");

        let tour_path = dir.join("run.tour");
        fs::write(
            &tour_path,
            "NAME : test\nTYPE : TOUR\nDIMENSION : 3\nTOUR_SECTION\n2\n1\n3\n-1\nEOF\n",
        )
        .expect("write tour file");

        let parsed = TsplibTour::parse_tsplib_tour(&tour_path).expect("parse tsplib tour");
        assert_eq!(parsed, vec![1, 0, 2]);

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn parse_tsplib_tour_is_permissive_without_dimension_header() {
        let dir = unique_temp_dir("parse-count");
        fs::create_dir_all(&dir).expect("create temp dir");

        let tour_path = dir.join("run.tour");
        fs::write(&tour_path, "TOUR_SECTION\n1\n-1\nEOF\n").expect("write tour file");

        let parsed =
            TsplibTour::parse_tsplib_tour(&tour_path).expect("permissive parsing should succeed");
        assert_eq!(parsed, vec![0]);

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn parse_tsplib_tour_skips_non_positive_non_terminator_ids() {
        let dir = unique_temp_dir("parse-non-positive");
        fs::create_dir_all(&dir).expect("create temp dir");

        let tour_path = dir.join("run.tour");
        fs::write(&tour_path, "TOUR_SECTION\n0\n-5\n2\n1\n-1\nEOF\n").expect("write tour file");

        let parsed =
            TsplibTour::parse_tsplib_tour(&tour_path).expect("permissive parsing should succeed");
        assert_eq!(parsed, vec![1, 0]);

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn parse_reads_headers_from_lkh_tour_output() {
        let text = r#"
NAME : problem.111984.tour
OPTIMUM = 111984
COMMENT : Length = 111984
COMMENT : Found by LKH-3 [Keld Helsgaun] Sun Feb  8 16:58:45 2026
TYPE : TOUR
DIMENSION : 4
TOUR_SECTION
1
3
2
4
-1
EOF
"#
        .to_string();

        let tour = TsplibTour::from_text(text).expect("parse tour");
        assert_eq!(tour.name.as_deref(), Some("problem.111984.tour"));
        assert_eq!(tour.comment_lines.len(), 2);
        assert_eq!(tour.tour_type, Some(TsplibTourType::Tour));
        assert_eq!(tour.dimension, Some(4));
        assert_eq!(tour.optimum, Some(111_984.0));
        assert_eq!(tour.tour_section, vec![1, 3, 2, 4]);
    }

    #[test]
    fn display_writes_tsplib_tour_format() {
        let mut tour = TsplibTour::new();
        tour.name = Some("sample.tour".to_string());
        tour.comment_lines.push("Length = 42".to_string());
        tour.tour_type = Some(TsplibTourType::Tour);
        tour.dimension = Some(3);
        tour.optimum = Some(42.0);
        tour.tour_section = vec![1, 2, 3];

        let text = tour.to_string();

        assert!(text.contains("NAME: sample.tour"));
        assert!(text.contains("TYPE: TOUR"));
        assert!(text.contains("COMMENT: Length = 42"));
        assert!(text.contains("DIMENSION: 3"));
        assert!(text.contains("OPTIMUM: 42"));
        assert!(text.contains("TOUR_SECTION\n1\n2\n3\n-1\n"));
        assert!(text.ends_with("EOF\n"));
    }
}
