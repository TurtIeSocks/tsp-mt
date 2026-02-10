use std::{fs, io::Read, path::Path};
use tsp_mt_derive::KvDisplay;

use crate::{Error, LKHNode, Result, SolverOptions};

/// Runtime input for LKH solver.
#[derive(Clone, Debug, KvDisplay)]
pub struct SolverInput {
    #[kv(fmt = "len")]
    pub(crate) nodes: Vec<LKHNode>,
}

impl SolverInput {
    pub fn new(points: &[LKHNode]) -> Self {
        Self {
            nodes: points.to_vec(),
        }
    }

    pub fn from_args(options: &SolverOptions) -> Result<Self> {
        let nodes = match options.input_path() {
            Some(path) => from_file(path)?,
            None => from_stdin()?,
        };
        Ok(Self { nodes })
    }

    pub fn points_len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn n(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn get_point(&self, idx: usize) -> LKHNode {
        self.nodes[idx]
    }
}

fn from_stdin() -> Result<Vec<LKHNode>> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    parse_points_from_tokens(&input)
}

fn from_file(path: &Path) -> Result<Vec<LKHNode>> {
    let bytes = fs::read(path)?;
    let input = std::str::from_utf8(&bytes).map_err(|_| {
        Error::invalid_input(format!(
            "Input file must be UTF-8 raw text: {}",
            path.display()
        ))
    })?;
    parse_points_from_rows(input, path)
}

fn parse_points_from_tokens(input: &str) -> Result<Vec<LKHNode>> {
    let mut points = Vec::new();
    for (idx, tok) in input.split_whitespace().enumerate() {
        points.push(parse_lat_lng(tok, &format!("Token {}", idx + 1))?);
    }

    if points.is_empty() {
        return Err(Error::invalid_input("No points provided on stdin."));
    }

    Ok(points)
}

fn parse_points_from_rows(input: &str, path: &Path) -> Result<Vec<LKHNode>> {
    let mut points = Vec::new();
    for (idx, line) in input.lines().enumerate() {
        let line_no = idx + 1;
        let row = line.trim_end_matches('\r');
        if row.trim().is_empty() {
            return Err(Error::invalid_input(format!(
                "Input file {} line {}: empty row; expected 'lat,lng'",
                path.display(),
                line_no
            )));
        }
        if row.split_whitespace().count() != 1 {
            return Err(Error::invalid_input(format!(
                "Input file {} line {}: expected exactly one 'lat,lng' row",
                path.display(),
                line_no
            )));
        }

        points.push(parse_lat_lng(
            row,
            &format!("Input file {} line {}", path.display(), line_no),
        )?);
    }

    if points.is_empty() {
        return Err(Error::invalid_input(format!(
            "Input file is empty: {}",
            path.display()
        )));
    }

    Ok(points)
}

fn parse_lat_lng(raw: &str, ctx: &str) -> Result<LKHNode> {
    let mut it = raw.split(',');
    let lat_s = it
        .next()
        .ok_or_else(|| Error::invalid_input(format!("{ctx}: missing latitude")))?;
    let lon_s = it
        .next()
        .ok_or_else(|| Error::invalid_input(format!("{ctx}: missing longitude")))?;

    if it.next().is_some() {
        return Err(Error::invalid_input(format!(
            "{ctx}: expected 'lat,lng' but got extra comma fields: {raw}"
        )));
    }

    let lat: f64 = lat_s
        .parse()
        .map_err(|_| Error::invalid_input(format!("{ctx}: invalid latitude: {lat_s}")))?;
    let lon: f64 = lon_s
        .parse()
        .map_err(|_| Error::invalid_input(format!("{ctx}: invalid longitude: {lon_s}")))?;

    let point = LKHNode::from_lat_lng(lat, lon);
    if !point.is_valid() {
        return Err(Error::invalid_input(format!(
            "{ctx}: invalid lat/lng values: {raw}"
        )));
    }

    Ok(point)
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::{from_file, parse_points_from_rows, parse_points_from_tokens};

    #[test]
    fn parse_points_from_tokens_parses_whitespace_separated_lat_lng_tokens() {
        let points = parse_points_from_tokens("1.0,2.0\n3.0,4.0 5.0,6.0").expect("parse points");
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].to_string(), "1.0,2.0");
        assert_eq!(points[2].to_string(), "5.0,6.0");
    }

    #[test]
    fn parse_points_from_tokens_rejects_empty_input() {
        let err = parse_points_from_tokens(" \n\t ").expect_err("empty input should fail");
        assert!(err.to_string().contains("No points provided on stdin."));
    }

    #[test]
    fn parse_points_from_tokens_rejects_extra_comma_fields() {
        let err = parse_points_from_tokens("1,2,3").expect_err("extra fields should fail");
        assert!(err.to_string().contains("expected 'lat,lng'"));
    }

    #[test]
    fn parse_points_from_tokens_rejects_non_numeric_coordinates() {
        let err = parse_points_from_tokens("a,2").expect_err("invalid latitude should fail");
        assert!(err.to_string().contains("invalid latitude"));
    }

    #[test]
    fn parse_points_from_tokens_rejects_out_of_range_coordinates() {
        let err = parse_points_from_tokens("91,0").expect_err("out of range latitude should fail");
        assert!(err.to_string().contains("invalid lat/lng values"));
    }

    #[test]
    fn parse_points_from_rows_parses_valid_lat_lng_rows() {
        let path = std::path::Path::new("points.txt");
        let points =
            parse_points_from_rows("1.0,2.0\n3.0,4.0\r\n5.0,6.0", path).expect("parse rows");
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].to_string(), "1.0,2.0");
        assert_eq!(points[2].to_string(), "5.0,6.0");
    }

    #[test]
    fn parse_points_from_rows_rejects_empty_rows() {
        let path = std::path::Path::new("points.txt");
        let err =
            parse_points_from_rows("1.0,2.0\n\n3.0,4.0", path).expect_err("blank line should fail");
        assert!(err.to_string().contains("empty row"));
    }

    #[test]
    fn parse_points_from_rows_rejects_whitespace_separated_tokens_on_one_line() {
        let path = std::path::Path::new("points.txt");
        let err = parse_points_from_rows("1.0,2.0 3.0,4.0", path)
            .expect_err("space-separated values in one line should fail");
        assert!(
            err.to_string()
                .contains("expected exactly one 'lat,lng' row")
        );
    }

    #[test]
    fn parse_points_from_rows_rejects_empty_file() {
        let path = std::path::Path::new("points.txt");
        let err = parse_points_from_rows("", path).expect_err("empty file should fail");
        assert!(err.to_string().contains("Input file is empty"));
    }

    #[test]
    fn parse_points_from_rows_rejects_out_of_range_coordinates() {
        let path = std::path::Path::new("points.txt");
        let err = parse_points_from_rows("91,0", path).expect_err("invalid row should fail");
        assert!(err.to_string().contains("invalid lat/lng values"));
    }

    #[test]
    fn from_file_rejects_non_utf8_content() {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("tsp-mt-non-utf8-{unique}.txt"));
        fs::write(&path, [0xFF, 0xFE, 0xFD]).expect("write test file");

        let err = from_file(&path).expect_err("non-utf8 file should fail");
        assert!(
            err.to_string()
                .contains("Input file must be UTF-8 raw text")
        );

        let _ = fs::remove_file(path);
    }
}
