use std::{
    fmt::{Display, Formatter},
    fs,
    path::Path,
};

use crate::{LkhResult, spec_writer::SpecWriter};
use lkh_derive::LkhDisplay;

const EUC2D_SCALE: f64 = 1_000.0;
const TSPLIB_NODE_ID_BASE: usize = 1;
const TSPLIB_SECTION_END_MARKER: isize = -1;

/// TSPLIB `TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum TsplibProblemType {
    Tsp,
    Atsp,
    Sop,
    Hcp,
    Cvrp,
    Tour,
}

/// TSPLIB `EDGE_WEIGHT_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
#[lkh(separator = "_")]
pub enum EdgeWeightType {
    Explicit,
    #[lkh("EUC_2D")]
    Euc2d,
    #[lkh("EUC_3D")]
    Euc3d,
    #[lkh("MAX_2D")]
    Max2d,
    #[lkh("MAX_3D")]
    Max3d,
    #[lkh("MAN_2D")]
    Man2d,
    #[lkh("MAN_3D")]
    Man3d,
    #[lkh("CEIL_2D")]
    Ceil2d,
    Geo,
    Att,
    Xray1,
    Xray2,
    Special,
}

/// TSPLIB `EDGE_WEIGHT_FORMAT` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
#[lkh(separator = "_")]
pub enum EdgeWeightFormat {
    Function,
    FullMatrix,
    UpperRow,
    LowerRow,
    UpperDiagRow,
    LowerDiagRow,
    UpperCol,
    LowerCol,
    UpperDiagCol,
    LowerDiagCol,
}

/// TSPLIB `EDGE_DATA_FORMAT` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
#[lkh(separator = "_")]
pub enum EdgeDataFormat {
    EdgeList,
    AdjList,
}

/// TSPLIB `NODE_COORD_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
#[lkh(separator = "_")]
pub enum NodeCoordType {
    TwodCoords,
    ThreedCoords,
    NoCoords,
}

/// TSPLIB `DISPLAY_DATA_TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
#[lkh(separator = "_")]
pub enum DisplayDataType {
    CoordDisplay,
    TwodDisplay,
    NoDisplay,
}

/// Entry in `NODE_COORD_SECTION`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeCoord {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
}

impl NodeCoord {
    pub const fn twod(id: usize, x: f64, y: f64) -> Self {
        Self { id, x, y, z: None }
    }

    pub const fn threed(id: usize, x: f64, y: f64, z: f64) -> Self {
        Self {
            id,
            x,
            y,
            z: Some(z),
        }
    }
}

impl Display for NodeCoord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.id, self.x, self.y)?;
        if let Some(z) = self.z {
            write!(f, " {}", z)?;
        }
        Ok(())
    }
}

/// Entry in `DISPLAY_DATA_SECTION`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayDataEntry {
    pub id: usize,
    pub x: f64,
    pub y: f64,
}

impl Display for DisplayDataEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.id, self.x, self.y)
    }
}

/// Entry in `DEMAND_SECTION`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DemandEntry {
    pub id: usize,
    pub demand: i64,
}

impl Display for DemandEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.demand)
    }
}

/// Entry in `FIXED_EDGES_SECTION`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedEdge {
    pub from: usize,
    pub to: usize,
}

impl Display for FixedEdge {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.from, self.to)
    }
}

/// Entry in `EDGE_DATA_SECTION`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EdgeDataEntry {
    Edge { from: usize, to: usize },
    AdjList { node: usize, neighbors: Vec<usize> },
}

impl Display for EdgeDataEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // let x: Vec<i64> = vec![1, 2, 3, 4];
        // let y = x.join(" ");
        match self {
            Self::Edge { from, to } => write!(f, "{from} {to}"),
            Self::AdjList { node, neighbors } => {
                write!(f, "{node}")?;
                for neighbor in neighbors {
                    write!(f, " {neighbor}")?;
                }
                write!(f, " {TSPLIB_SECTION_END_MARKER}")
            }
        }
    }
}

/// Full TSPLIB problem model used by LKH.
#[derive(Clone, Debug, PartialEq)]
pub struct TsplibProblem {
    pub name: String,
    pub problem_type: TsplibProblemType,
    pub comment_lines: Vec<String>,
    pub dimension: Option<usize>,
    pub capacity: Option<i64>,
    pub edge_weight_type: Option<EdgeWeightType>,
    pub edge_weight_format: Option<EdgeWeightFormat>,
    pub edge_data_format: Option<EdgeDataFormat>,
    pub node_coord_type: Option<NodeCoordType>,
    pub display_data_type: Option<DisplayDataType>,
    pub edge_weight_section: Vec<Vec<i64>>,
    pub edge_data_section: Vec<EdgeDataEntry>,
    pub node_coord_section: Vec<NodeCoord>,
    pub display_data_section: Vec<DisplayDataEntry>,
    pub fixed_edges_section: Vec<FixedEdge>,
    pub demand_section: Vec<DemandEntry>,
    pub depot_section: Vec<usize>,
    pub tour_section: Vec<usize>,
    pub emit_eof: bool,
}

impl TsplibProblem {
    pub fn new(name: impl Into<String>, problem_type: TsplibProblemType) -> Self {
        Self {
            name: name.into(),
            problem_type,
            comment_lines: Vec::new(),
            dimension: None,
            capacity: None,
            edge_weight_type: None,
            edge_weight_format: None,
            edge_data_format: None,
            node_coord_type: None,
            display_data_type: None,
            edge_weight_section: Vec::new(),
            edge_data_section: Vec::new(),
            node_coord_section: Vec::new(),
            display_data_section: Vec::new(),
            fixed_edges_section: Vec::new(),
            demand_section: Vec::new(),
            depot_section: Vec::new(),
            tour_section: Vec::new(),
            emit_eof: true,
        }
    }

    pub fn from_euc2d_points<I>(name: impl Into<String>, points: I) -> Self
    where
        I: IntoIterator<Item = (f64, f64)>,
        I::IntoIter: ExactSizeIterator,
    {
        let points = points.into_iter();

        let mut problem = Self::new(name, TsplibProblemType::Tsp);
        problem.dimension = Some(points.len());
        problem.edge_weight_type = Some(EdgeWeightType::Euc2d);
        problem.node_coord_type = Some(NodeCoordType::TwodCoords);

        for (idx, (x, y)) in points.enumerate() {
            problem.node_coord_section.push(NodeCoord::twod(
                idx + TSPLIB_NODE_ID_BASE,
                (x * EUC2D_SCALE).round(),
                (y * EUC2D_SCALE).round(),
            ));
        }

        problem
    }

    pub fn write_to_file(&self, path: &Path) -> LkhResult<()> {
        fs::write(path, self.to_string())?;
        Ok(())
    }
}

impl Display for TsplibProblem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = SpecWriter::new(f);

        writer.kv_colon("NAME", &self.name)?;
        writer.kv_colon("TYPE", self.problem_type)?;

        for comment in &self.comment_lines {
            writer.kv_colon("COMMENT", comment)?;
        }

        writer.opt_kv_colon("DIMENSION", self.dimension)?;
        writer.opt_kv_colon("CAPACITY", self.capacity)?;
        writer.opt_kv_colon("EDGE_WEIGHT_TYPE", self.edge_weight_type)?;
        writer.opt_kv_colon("EDGE_WEIGHT_FORMAT", self.edge_weight_format)?;
        writer.opt_kv_colon("EDGE_DATA_FORMAT", self.edge_data_format)?;
        writer.opt_kv_colon("NODE_COORD_TYPE", self.node_coord_type)?;
        writer.opt_kv_colon("DISPLAY_DATA_TYPE", self.display_data_type)?;
        writer.lines("NODE_COORD_SECTION", &self.node_coord_section)?;

        if !self.depot_section.is_empty() {
            writer.lines("DEPOT_SECTION", &self.depot_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        writer.lines("DEMAND_SECTION", &self.demand_section)?;

        if !self.edge_data_section.is_empty() {
            writer.lines("EDGE_DATA_SECTION", &self.edge_data_section)?;
            if self.edge_data_format == Some(EdgeDataFormat::EdgeList) {
                writer.line(TSPLIB_SECTION_END_MARKER)?;
            }
        }

        writer.lines("FIXED_EDGES_SECTION", &self.fixed_edges_section)?;
        writer.lines("DISPLAY_DATA_SECTION", &self.display_data_section)?;

        if !self.tour_section.is_empty() {
            writer.lines("TOUR_SECTION", &self.tour_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        if !self.edge_weight_section.is_empty() {
            writer.line("EDGE_WEIGHT_SECTION")?;
            for row in &self.edge_weight_section {
                writer.row(row)?;
            }
        }

        if self.emit_eof {
            writer.line("EOF")?;
        }

        Ok(())
    }
}

pub struct TsplibProblemWriter;

impl TsplibProblemWriter {
    pub fn write(path: &Path, problem: &TsplibProblem) -> LkhResult<()> {
        problem.write_to_file(path)
    }

    pub fn write_euc2d<I>(path: &Path, name: &str, points: I) -> LkhResult<()>
    where
        I: IntoIterator<Item = (f64, f64)>,
        I::IntoIter: ExactSizeIterator,
    {
        let problem = TsplibProblem::from_euc2d_points(name, points);
        problem.write_to_file(path)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        DisplayDataType, EdgeDataEntry, EdgeDataFormat, EdgeWeightFormat, EdgeWeightType,
        NodeCoord, NodeCoordType, TsplibProblem, TsplibProblemType, TsplibProblemWriter,
    };

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("lkh-tests-{name}-{nanos}"))
    }

    #[test]
    fn display_emits_header_fields_and_sections() {
        let mut problem = TsplibProblem::new("sample", TsplibProblemType::Tsp);
        problem.comment_lines.push("first".to_string());
        problem.comment_lines.push("second".to_string());
        problem.dimension = Some(3);
        problem.capacity = Some(99);
        problem.edge_weight_type = Some(EdgeWeightType::Euc2d);
        problem.edge_weight_format = Some(EdgeWeightFormat::FullMatrix);
        problem.edge_data_format = Some(EdgeDataFormat::AdjList);
        problem.node_coord_type = Some(NodeCoordType::TwodCoords);
        problem.display_data_type = Some(DisplayDataType::CoordDisplay);
        problem.node_coord_section = vec![
            NodeCoord::twod(1, 10.0, 20.0),
            NodeCoord::threed(2, 30.0, 40.0, 50.0),
        ];
        problem.edge_data_section = vec![EdgeDataEntry::AdjList {
            node: 1,
            neighbors: vec![2, 3],
        }];
        problem.tour_section = vec![1, 3, 2];
        problem.edge_weight_section = vec![vec![0, 7, 3], vec![7, 0, 5], vec![3, 5, 0]];

        let text = problem.to_string();
        assert!(text.contains("NAME: sample"));
        assert!(text.contains("TYPE: TSP"));
        assert!(text.contains("COMMENT: first"));
        assert!(text.contains("DIMENSION: 3"));
        assert!(text.contains("CAPACITY: 99"));
        assert!(text.contains("EDGE_WEIGHT_TYPE: EUC_2D"));
        assert!(text.contains("EDGE_WEIGHT_FORMAT: FULL_MATRIX"));
        assert!(text.contains("EDGE_DATA_FORMAT: ADJ_LIST"));
        assert!(text.contains("NODE_COORD_TYPE: TWOD_COORDS"));
        assert!(text.contains("DISPLAY_DATA_TYPE: COORD_DISPLAY"));
        assert!(text.contains("NODE_COORD_SECTION\n1 10 20\n2 30 40 50\n"));
        assert!(text.contains("EDGE_DATA_SECTION\n1 2 3 -1\n"));
        assert!(text.contains("TOUR_SECTION\n1\n3\n2\n-1\n"));
        assert!(text.contains("EDGE_WEIGHT_SECTION\n0 7 3\n7 0 5\n3 5 0\n"));
        assert!(text.ends_with("EOF\n"));
    }

    #[test]
    fn display_keeps_headers_before_sections() {
        let mut problem = TsplibProblem::new("sample", TsplibProblemType::Atsp);
        problem.dimension = Some(2);
        problem.edge_weight_type = Some(EdgeWeightType::Explicit);
        problem.edge_weight_section = vec![vec![0, 1], vec![2, 0]];

        let text = problem.to_string();
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(lines[0], "NAME: sample");
        assert_eq!(lines[1], "TYPE: ATSP");
        assert_eq!(lines[2], "DIMENSION: 2");
        assert_eq!(lines[3], "EDGE_WEIGHT_TYPE: EXPLICIT");
        assert_eq!(lines[4], "EDGE_WEIGHT_SECTION");
    }

    #[test]
    fn write_euc2d_matches_legacy_rounding_and_ids() {
        let dir = unique_temp_dir("problem-writer");
        fs::create_dir_all(&dir).expect("create temp dir");

        let problem_path = dir.join("problem.tsp");
        TsplibProblemWriter::write_euc2d(
            &problem_path,
            "sample",
            vec![(0.1234, 0.5678), (1.2, 1.8), (9.9994, 0.0004)],
        )
        .expect("write problem file");

        let text = fs::read_to_string(&problem_path).expect("read problem file");
        assert!(text.contains("NAME: sample"));
        assert!(text.contains("TYPE: TSP"));
        assert!(text.contains("DIMENSION: 3"));
        assert!(text.contains("EDGE_WEIGHT_TYPE: EUC_2D"));
        assert!(text.contains("NODE_COORD_TYPE: TWOD_COORDS"));
        assert!(text.contains("NODE_COORD_SECTION\n1 123 568\n2 1200 1800\n3 9999 0\n"));
        assert!(text.ends_with("EOF\n"));

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }
}
