use std::{fs, path::Path, process::Output};

use crate::{LkhError, LkhResult};

const TOUR_SECTION_HEADER: &str = "TOUR_SECTION";
const TOUR_END_MARKER: &str = "-1";
const EOF_MARKER: &str = "EOF";
const MIN_VALID_TSPLIB_NODE_ID: isize = 1;
const TSPLIB_NODE_ID_OFFSET: usize = 1;

pub struct LkhProcess;

impl LkhProcess {
    pub fn ensure_success(context: &str, out: &Output) -> LkhResult<()> {
        if out.status.success() {
            return Ok(());
        }

        Err(LkhError::ProcessFailed {
            context: context.to_string(),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        })
    }

    pub fn parse_tsplib_tour(path: &Path, n: usize) -> LkhResult<Vec<usize>> {
        let text = fs::read_to_string(path)?;
        let mut in_section = false;
        let mut tour: Vec<usize> = Vec::with_capacity(n);

        for line in text.lines() {
            let line = line.trim();
            if line.eq_ignore_ascii_case(TOUR_SECTION_HEADER) {
                in_section = true;
                continue;
            }
            if !in_section {
                continue;
            }
            if line == TOUR_END_MARKER || line.eq_ignore_ascii_case(EOF_MARKER) {
                break;
            }
            let id: isize = line
                .parse()
                .map_err(|e| LkhError::invalid_data(format!("Bad tour line '{line}': {e}")))?;
            if id < MIN_VALID_TSPLIB_NODE_ID {
                continue;
            }
            tour.push((id as usize) - TSPLIB_NODE_ID_OFFSET);
        }

        if tour.len() != n {
            return Err(LkhError::invalid_data(format!(
                "Expected {n} nodes in tour, got {}",
                tour.len()
            )));
        }

        Ok(tour)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::LkhProcess;

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
        fs::write(&tour_path, "NAME : test\nTOUR_SECTION\n2\n1\n3\n-1\nEOF\n")
            .expect("write tour file");

        let parsed = LkhProcess::parse_tsplib_tour(&tour_path, 3).expect("parse tsplib tour");
        assert_eq!(parsed, vec![1, 0, 2]);

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn parse_tsplib_tour_errors_on_wrong_node_count() {
        let dir = unique_temp_dir("parse-count");
        fs::create_dir_all(&dir).expect("create temp dir");

        let tour_path = dir.join("run.tour");
        fs::write(&tour_path, "TOUR_SECTION\n1\n-1\nEOF\n").expect("write tour file");

        let err =
            LkhProcess::parse_tsplib_tour(&tour_path, 2).expect_err("expected node-count mismatch");
        let msg = err.to_string();
        assert!(msg.contains("Expected 2 nodes in tour, got 1"));

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }
}
