use std::io::Read;
use tsp_mt_derive::KvDisplay;

use crate::{Error, LKHNode, Result};

/// Runtime input for LKH solver.
#[derive(Clone, Debug, KvDisplay)]
pub struct SolverInput {
    #[kv(fmt = "len")]
    pub(crate) points: Vec<LKHNode>,
}

impl SolverInput {
    pub fn new(points: &[LKHNode]) -> Self {
        Self {
            points: points.to_vec(),
        }
    }

    pub fn from_args() -> Result<Self> {
        Ok(Self {
            points: read_points_from_stdin()?,
        })
    }

    pub fn points_len(&self) -> usize {
        self.points.len()
    }

    pub(crate) fn n(&self) -> usize {
        self.points.len()
    }

    pub(crate) fn get_point(&self, idx: usize) -> LKHNode {
        self.points[idx]
    }
}

fn read_points_from_stdin() -> Result<Vec<LKHNode>> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    parse_points(&input)
}

fn parse_points(input: &str) -> Result<Vec<LKHNode>> {
    let mut points = Vec::new();
    for (idx, tok) in input.split_whitespace().enumerate() {
        let mut it = tok.split(',');
        let lat_s = it
            .next()
            .ok_or_else(|| Error::invalid_input(format!("Token {}: missing latitude", idx + 1)))?;
        let lon_s = it
            .next()
            .ok_or_else(|| Error::invalid_input(format!("Token {}: missing longitude", idx + 1)))?;

        if it.next().is_some() {
            return Err(Error::invalid_input(format!(
                "Token {}: expected 'lat,lng' but got extra comma fields: {tok}",
                idx + 1
            )));
        }

        let lat: f64 = lat_s.parse().map_err(|_| {
            Error::invalid_input(format!("Token {}: invalid latitude: {}", idx + 1, lat_s))
        })?;
        let lon: f64 = lon_s.parse().map_err(|_| {
            Error::invalid_input(format!("Token {}: invalid longitude: {}", idx + 1, lon_s))
        })?;

        points.push(LKHNode::from_lat_lng(lat, lon));
    }

    if points.is_empty() {
        return Err(Error::invalid_input("No points provided on stdin."));
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::parse_points;

    #[test]
    fn parse_points_parses_whitespace_separated_lat_lng_tokens() {
        let points = parse_points("1.0,2.0\n3.0,4.0 5.0,6.0").expect("parse points");
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].to_string(), "1.0,2.0");
        assert_eq!(points[2].to_string(), "5.0,6.0");
    }

    #[test]
    fn parse_points_rejects_empty_input() {
        let err = parse_points(" \n\t ").expect_err("empty input should fail");
        assert!(err.to_string().contains("No points provided on stdin."));
    }

    #[test]
    fn parse_points_rejects_extra_comma_fields() {
        let err = parse_points("1,2,3").expect_err("extra fields should fail");
        assert!(err.to_string().contains("expected 'lat,lng'"));
    }

    #[test]
    fn parse_points_rejects_non_numeric_coordinates() {
        let err = parse_points("a,2").expect_err("invalid latitude should fail");
        assert!(err.to_string().contains("invalid latitude"));
    }
}
