//! TSPLIB problem-file model and writer.
//!
//! For the TSPLIB fields and sections supported by this crate, see:
//! `crates/lkh/docs/TSPLIB_SPEC.md`.

use std::{
    fmt::{Display, Formatter},
    fs,
    path::PathBuf,
};

use crate::{LkhError, LkhResult, spec_writer::SpecWriter, with_methods_error};
use lkh_derive::{LkhDisplay, WithMethods};

const TSPLIB_NODE_ID_BASE: usize = 1;
const TSPLIB_SECTION_END_MARKER: isize = -1;

/// TSPLIB/LKH `TYPE` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, LkhDisplay)]
pub enum TsplibProblemType {
    Tsp,
    Atsp,
    Sop,
    Hcp,
    Hpp,
    Bwtsp,
    Cluvrp,
    Ccvrp,
    Cvrp,
    Acvrp,
    Adcvrp,
    Cvrptw,
    Ktsp,
    Mlp,
    Msctsp,
    Ovrp,
    Pctsp,
    Pdptw,
    Pdtsp,
    Pdtspf,
    Pdtspl,
    Ptsp,
    Trp,
    Rctvrp,
    Rctvrptw,
    Softcluvrp,
    Sttsp,
    Tsptw,
    Vrpb,
    Vrpbtw,
    Vrppd,
    #[lkh("1-PDTSP")]
    OnePdtsp,
    #[lkh("M-PDTSP")]
    MPdtsp,
    #[lkh("M1-PDTSP")]
    M1Pdtsp,
    Tspdl,
    Ctsp,
    #[lkh("CTSP-D")]
    CtspD,
    Gctsp,
    Ccctsp,
    Cbtsp,
    #[lkh("CBNTSP")]
    CbnTsp,
    Tour,
}

/// TSPLIB/LKH `EDGE_WEIGHT_TYPE` values.
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
    #[lkh("CEIL_3D")]
    Ceil3d,
    #[lkh("EXACT_2D")]
    Exact2d,
    #[lkh("EXACT_3D")]
    Exact3d,
    #[lkh("FLOOR_2D")]
    Floor2d,
    #[lkh("FLOOR_3D")]
    Floor3d,
    Geo,
    Geom,
    #[lkh("GEO_MEEUS")]
    GeoMeeus,
    #[lkh("GEOM_MEEUS")]
    GeomMeeus,
    #[lkh("TOR_2D")]
    Tor2d,
    #[lkh("TOR_3D")]
    Tor3d,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DemandEntry {
    pub id: usize,
    pub demands: Vec<i64>,
}

impl DemandEntry {
    pub fn single(id: usize, demand: i64) -> Self {
        Self {
            id,
            demands: vec![demand],
        }
    }

    pub fn multi(id: usize, demands: Vec<i64>) -> Self {
        Self { id, demands }
    }
}

impl Display for DemandEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)?;
        for demand in &self.demands {
            write!(f, " {}", demand)?;
        }
        Ok(())
    }
}

/// Entry in `DRAFT_LIMIT_SECTION`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftLimitEntry {
    pub id: usize,
    pub draft_limit: i64,
}

impl Display for DraftLimitEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.draft_limit)
    }
}

/// Entry in `SERVICE_TIME_SECTION`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ServiceTimeEntry {
    pub id: usize,
    pub service_time: f64,
}

impl Display for ServiceTimeEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.service_time)
    }
}

/// Entry in `TIME_WINDOW_SECTION`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeWindowEntry {
    pub id: usize,
    pub earliest: f64,
    pub latest: f64,
}

impl Display for TimeWindowEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.id, self.earliest, self.latest)
    }
}

/// Entry in `PICKUP_AND_DELIVERY_SECTION`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickupAndDeliveryEntry {
    pub id: usize,
    pub demand: i64,
    pub earliest: f64,
    pub latest: f64,
    pub service_time: f64,
    pub pickup: i64,
    pub delivery: i64,
}

impl Display for PickupAndDeliveryEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {} {} {} {} {}",
            self.id,
            self.demand,
            self.earliest,
            self.latest,
            self.service_time,
            self.pickup,
            self.delivery
        )
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
    Edge {
        from: usize,
        to: usize,
        weight: Option<i64>,
    },
    AdjList {
        node: usize,
        neighbors: Vec<usize>,
    },
}

impl Display for EdgeDataEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Edge { from, to, weight } => {
                write!(f, "{from} {to}")?;
                if let Some(weight) = weight {
                    write!(f, " {weight}")?;
                }
                Ok(())
            }
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

/// Generic `<id> <members...> -1` entry used by several LKH sections.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetSectionEntry {
    pub id: usize,
    pub members: Vec<usize>,
}

impl Display for SetSectionEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)?;
        for member in &self.members {
            write!(f, " {member}")?;
        }
        write!(f, " {TSPLIB_SECTION_END_MARKER}")
    }
}

/// Full TSPLIB problem model used by LKH.
#[derive(Clone, Debug, PartialEq, WithMethods)]
#[with(error = LkhError)]
pub struct TsplibProblem {
    pub name: String,
    pub problem_type: TsplibProblemType,
    pub comment_lines: Vec<String>,
    pub dimension: Option<usize>,
    pub capacity: Option<i64>,
    pub distance: Option<f64>,
    pub edge_weight_type: Option<EdgeWeightType>,
    pub edge_weight_format: Option<EdgeWeightFormat>,
    pub edge_data_format: Option<EdgeDataFormat>,
    pub node_coord_type: Option<NodeCoordType>,
    pub display_data_type: Option<DisplayDataType>,
    pub demand_dimension: Option<usize>,
    pub grid_size: Option<f64>,
    pub groups: Option<usize>,
    pub gvrp_sets: Option<usize>,
    pub relaxation_level: Option<usize>,
    pub risk_threshold: Option<i64>,
    pub salesmen: Option<usize>,
    pub vehicles: Option<usize>,
    pub scale: Option<i64>,
    pub service_time: Option<f64>,
    pub edge_weight_section: Vec<Vec<i64>>,
    pub edge_data_section: Vec<EdgeDataEntry>,
    pub node_coord_section: Vec<NodeCoord>,
    pub display_data_section: Vec<DisplayDataEntry>,
    pub fixed_edges_section: Vec<FixedEdge>,
    pub demand_section: Vec<DemandEntry>,
    pub depot_section: Vec<usize>,
    pub tour_section: Vec<usize>,
    pub backhaul_section: Vec<usize>,
    pub ctsp_set_section: Vec<SetSectionEntry>,
    pub draft_limit_section: Vec<DraftLimitEntry>,
    pub gctsp_section: Vec<Vec<i64>>,
    pub gctsp_set_section: Vec<SetSectionEntry>,
    pub group_section: Vec<SetSectionEntry>,
    pub gvrp_set_section: Vec<SetSectionEntry>,
    pub pickup_and_delivery_section: Vec<PickupAndDeliveryEntry>,
    pub required_nodes_section: Vec<usize>,
    pub service_time_section: Vec<ServiceTimeEntry>,
    pub time_window_section: Vec<TimeWindowEntry>,
    pub emit_eof: bool,
}

with_methods_error!(TsplibProblemWithMethodsError);

impl TsplibProblem {
    /// Creates an empty TSPLIB/LKH problem model for the given `TYPE`.
    pub fn new(problem_type: TsplibProblemType) -> Self {
        Self {
            name: "PROBLEM".to_string(),
            problem_type,
            comment_lines: Vec::new(),
            dimension: None,
            capacity: None,
            distance: None,
            edge_weight_type: None,
            edge_weight_format: None,
            edge_data_format: None,
            node_coord_type: None,
            display_data_type: None,
            demand_dimension: None,
            grid_size: None,
            groups: None,
            gvrp_sets: None,
            relaxation_level: None,
            risk_threshold: None,
            salesmen: None,
            vehicles: None,
            scale: None,
            service_time: None,
            edge_weight_section: Vec::new(),
            edge_data_section: Vec::new(),
            node_coord_section: Vec::new(),
            display_data_section: Vec::new(),
            fixed_edges_section: Vec::new(),
            demand_section: Vec::new(),
            depot_section: Vec::new(),
            tour_section: Vec::new(),
            backhaul_section: Vec::new(),
            ctsp_set_section: Vec::new(),
            draft_limit_section: Vec::new(),
            gctsp_section: Vec::new(),
            gctsp_set_section: Vec::new(),
            group_section: Vec::new(),
            gvrp_set_section: Vec::new(),
            pickup_and_delivery_section: Vec::new(),
            required_nodes_section: Vec::new(),
            service_time_section: Vec::new(),
            time_window_section: Vec::new(),
            emit_eof: true,
        }
    }

    /// Convenience constructor for Euclidean 2D TSP instances.
    ///
    /// It sets:
    /// - `TYPE = TSP`
    /// - `EDGE_WEIGHT_TYPE = EUC_2D`
    /// - `NODE_COORD_TYPE = TWOD_COORDS`
    /// - `DIMENSION = points.len()`
    /// - `NODE_COORD_SECTION` using 1-based node ids
    pub fn from_euc2d_points<I>(points: I) -> Self
    where
        I: IntoIterator<Item = (f64, f64)>,
        I::IntoIter: ExactSizeIterator,
    {
        let points = points.into_iter();

        let mut problem = Self::new(TsplibProblemType::Tsp);
        problem.dimension = Some(points.len());
        problem.edge_weight_type = Some(EdgeWeightType::Euc2d);
        problem.node_coord_type = Some(NodeCoordType::TwodCoords);

        for (idx, (x, y)) in points.enumerate() {
            problem
                .node_coord_section
                .push(NodeCoord::twod(idx + TSPLIB_NODE_ID_BASE, x, y));
        }

        problem
    }

    /// Serializes and writes this problem file to disk.
    pub fn write_to_file(&self, file_path: impl Into<PathBuf>) -> LkhResult<()> {
        fs::write(file_path.into(), self.to_string()).map_err(LkhError::Io)
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
        writer.opt_kv_colon("DISTANCE", self.distance)?;
        writer.opt_kv_colon("EDGE_WEIGHT_TYPE", self.edge_weight_type)?;
        writer.opt_kv_colon("EDGE_WEIGHT_FORMAT", self.edge_weight_format)?;
        writer.opt_kv_colon("EDGE_DATA_FORMAT", self.edge_data_format)?;
        writer.opt_kv_colon("NODE_COORD_TYPE", self.node_coord_type)?;
        writer.opt_kv_colon("DISPLAY_DATA_TYPE", self.display_data_type)?;
        writer.opt_kv_colon("DEMAND_DIMENSION", self.demand_dimension)?;
        writer.opt_kv_colon("GRID_SIZE", self.grid_size)?;
        writer.opt_kv_colon("GROUPS", self.groups)?;
        writer.opt_kv_colon("GVRP_SETS", self.gvrp_sets)?;
        writer.opt_kv_colon("RELAXATION_LEVEL", self.relaxation_level)?;
        writer.opt_kv_colon("RISK_THRESHOLD", self.risk_threshold)?;
        writer.opt_kv_colon("SALESMEN", self.salesmen)?;
        writer.opt_kv_colon("VEHICLES", self.vehicles)?;
        writer.opt_kv_colon("SCALE", self.scale)?;
        writer.opt_kv_colon("SERVICE_TIME", self.service_time)?;

        writer.lines("NODE_COORD_SECTION", &self.node_coord_section)?;

        if !self.edge_data_section.is_empty() {
            writer.lines("EDGE_DATA_SECTION", &self.edge_data_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        if !self.fixed_edges_section.is_empty() {
            writer.lines("FIXED_EDGES_SECTION", &self.fixed_edges_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        writer.lines("DISPLAY_DATA_SECTION", &self.display_data_section)?;

        if !self.edge_weight_section.is_empty() {
            writer.line("EDGE_WEIGHT_SECTION")?;
            for row in &self.edge_weight_section {
                writer.row(row)?;
            }
        }

        if !self.tour_section.is_empty() {
            writer.lines("TOUR_SECTION", &self.tour_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        if !self.backhaul_section.is_empty() {
            writer.lines("BACKHAUL_SECTION", &self.backhaul_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        writer.lines("CTSP_SET_SECTION", &self.ctsp_set_section)?;
        writer.lines("DEMAND_SECTION", &self.demand_section)?;

        if !self.depot_section.is_empty() {
            writer.lines("DEPOT_SECTION", &self.depot_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        writer.lines("DRAFT_LIMIT_SECTION", &self.draft_limit_section)?;

        if !self.gctsp_section.is_empty() {
            writer.line("GCTSP_SECTION")?;
            for row in &self.gctsp_section {
                writer.row(row)?;
            }
        }

        writer.lines("GCTSP_SET_SECTION", &self.gctsp_set_section)?;
        writer.lines("GROUP_SECTION", &self.group_section)?;
        writer.lines("GVRP_SET_SECTION", &self.gvrp_set_section)?;
        writer.lines(
            "PICKUP_AND_DELIVERY_SECTION",
            &self.pickup_and_delivery_section,
        )?;

        if !self.required_nodes_section.is_empty() {
            writer.lines("REQUIRED_NODES_SECTION", &self.required_nodes_section)?;
            writer.line(TSPLIB_SECTION_END_MARKER)?;
        }

        writer.lines("SERVICE_TIME_SECTION", &self.service_time_section)?;
        writer.lines("TIME_WINDOW_SECTION", &self.time_window_section)?;

        if self.emit_eof {
            writer.line("EOF")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::{
        DemandEntry, DisplayDataType, DraftLimitEntry, EdgeDataEntry, EdgeDataFormat,
        EdgeWeightFormat, EdgeWeightType, NodeCoord, NodeCoordType, PickupAndDeliveryEntry,
        ServiceTimeEntry, SetSectionEntry, TimeWindowEntry, TsplibProblem, TsplibProblemType,
    };

    #[test]
    fn display_emits_lkh3_headers_and_sections() {
        let mut problem = TsplibProblem::new(TsplibProblemType::CtspD);
        problem.comment_lines.push("first".to_string());
        problem.comment_lines.push("second".to_string());
        problem.dimension = Some(3);
        problem.capacity = Some(99);
        problem.distance = Some(123.5);
        problem.edge_weight_type = Some(EdgeWeightType::Exact2d);
        problem.edge_weight_format = Some(EdgeWeightFormat::FullMatrix);
        problem.edge_data_format = Some(EdgeDataFormat::EdgeList);
        problem.node_coord_type = Some(NodeCoordType::TwodCoords);
        problem.display_data_type = Some(DisplayDataType::CoordDisplay);
        problem.demand_dimension = Some(2);
        problem.grid_size = Some(1000.0);
        problem.groups = Some(2);
        problem.gvrp_sets = Some(2);
        problem.relaxation_level = Some(1);
        problem.risk_threshold = Some(10);
        problem.salesmen = Some(2);
        problem.vehicles = Some(2);
        problem.scale = Some(100);
        problem.service_time = Some(0.5);
        problem.node_coord_section = vec![
            NodeCoord::twod(1, 10.0, 20.0),
            NodeCoord::threed(2, 30.0, 40.0, 50.0),
        ];
        problem.edge_data_section = vec![EdgeDataEntry::Edge {
            from: 1,
            to: 2,
            weight: Some(7),
        }];
        problem.fixed_edges_section = vec![super::FixedEdge { from: 1, to: 2 }];
        problem.display_data_section = vec![super::DisplayDataEntry {
            id: 1,
            x: 11.0,
            y: 22.0,
        }];
        problem.edge_weight_section = vec![vec![0, 7, 3], vec![7, 0, 5], vec![3, 5, 0]];
        problem.tour_section = vec![1, 3, 2];
        problem.backhaul_section = vec![2];
        problem.ctsp_set_section = vec![SetSectionEntry {
            id: 1,
            members: vec![2, 3],
        }];
        problem.demand_section = vec![DemandEntry::single(1, 0), DemandEntry::multi(2, vec![4, 5])];
        problem.depot_section = vec![1];
        problem.draft_limit_section = vec![DraftLimitEntry {
            id: 1,
            draft_limit: 6,
        }];
        problem.gctsp_section = vec![vec![1, 0, 1], vec![0, 1, 1]];
        problem.gctsp_set_section = vec![SetSectionEntry {
            id: 1,
            members: vec![1, 2],
        }];
        problem.group_section = vec![SetSectionEntry {
            id: 1,
            members: vec![2],
        }];
        problem.gvrp_set_section = vec![SetSectionEntry {
            id: 1,
            members: vec![1, 3],
        }];
        problem.pickup_and_delivery_section = vec![PickupAndDeliveryEntry {
            id: 1,
            demand: -3,
            earliest: 1.0,
            latest: 9.0,
            service_time: 0.5,
            pickup: 0,
            delivery: 2,
        }];
        problem.required_nodes_section = vec![1, 2];
        problem.service_time_section = vec![ServiceTimeEntry {
            id: 1,
            service_time: 0.25,
        }];
        problem.time_window_section = vec![TimeWindowEntry {
            id: 1,
            earliest: 0.0,
            latest: 10.0,
        }];

        let text = problem.to_string();

        assert!(text.contains("NAME: PROBLEM"));
        assert!(text.contains("TYPE: CTSP-D"));
        assert!(text.contains("COMMENT: first"));
        assert!(text.contains("DISTANCE: 123.5"));
        assert!(text.contains("EDGE_WEIGHT_TYPE: EXACT_2D"));
        assert!(text.contains("DEMAND_DIMENSION: 2"));
        assert!(text.contains("GVRP_SETS: 2"));
        assert!(text.contains("RELAXATION_LEVEL: 1"));
        assert!(text.contains("RISK_THRESHOLD: 10"));
        assert!(text.contains("SALESMEN: 2"));
        assert!(text.contains("VEHICLES: 2"));
        assert!(text.contains("SCALE: 100"));
        assert!(text.contains("SERVICE_TIME: 0.5"));

        assert!(text.contains("EDGE_DATA_SECTION\n1 2 7\n-1\n"));
        assert!(text.contains("FIXED_EDGES_SECTION\n1 2\n-1\n"));
        assert!(text.contains("TOUR_SECTION\n1\n3\n2\n-1\n"));
        assert!(text.contains("BACKHAUL_SECTION\n2\n-1\n"));
        assert!(text.contains("CTSP_SET_SECTION\n1 2 3 -1\n"));
        assert!(text.contains("DEMAND_SECTION\n1 0\n2 4 5\n"));
        assert!(text.contains("DEPOT_SECTION\n1\n-1\n"));
        assert!(text.contains("DRAFT_LIMIT_SECTION\n1 6\n"));
        assert!(text.contains("GCTSP_SECTION\n1 0 1\n0 1 1\n"));
        assert!(text.contains("GCTSP_SET_SECTION\n1 1 2 -1\n"));
        assert!(text.contains("GROUP_SECTION\n1 2 -1\n"));
        assert!(text.contains("GVRP_SET_SECTION\n1 1 3 -1\n"));
        assert!(text.contains("PICKUP_AND_DELIVERY_SECTION\n1 -3 1 9 0.5 0 2\n"));
        assert!(text.contains("REQUIRED_NODES_SECTION\n1\n2\n-1\n"));
        assert!(text.contains("SERVICE_TIME_SECTION\n1 0.25\n"));
        assert!(text.contains("TIME_WINDOW_SECTION\n1 0 10\n"));
        assert!(text.ends_with("EOF\n"));
    }

    #[test]
    fn display_keeps_headers_before_sections() {
        let mut problem = TsplibProblem::new(TsplibProblemType::Atsp);
        problem.dimension = Some(2);
        problem.edge_weight_type = Some(EdgeWeightType::Explicit);
        problem.edge_weight_section = vec![vec![0, 1], vec![2, 0]];

        let text = problem.to_string();
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(lines[0], "NAME: PROBLEM");
        assert_eq!(lines[1], "TYPE: ATSP");
        assert_eq!(lines[2], "DIMENSION: 2");
        assert_eq!(lines[3], "EDGE_WEIGHT_TYPE: EXPLICIT");
        assert_eq!(lines[4], "EDGE_WEIGHT_SECTION");
    }
}
