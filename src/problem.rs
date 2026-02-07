use std::{fs, io, path::Path};

use crate::node::LKHNode;

const EUC2D_SCALE: f64 = 1_000.0;
const TSPLIB_NODE_ID_BASE: usize = 1;
const TSPLIB_TYPE_TSP_LINE: &str = "TYPE: TSP\n";
const TSPLIB_EDGE_WEIGHT_TYPE_LINE: &str = "EDGE_WEIGHT_TYPE: EUC_2D\n";
const TSPLIB_NODE_COORD_SECTION_LINE: &str = "NODE_COORD_SECTION\n";
const TSPLIB_EOF_LINE: &str = "EOF\n";

pub(crate) struct TsplibProblemWriter;

impl TsplibProblemWriter {
    pub(crate) fn write_euc2d(path: &Path, name: &str, points: &[LKHNode]) -> io::Result<()> {
        let mut body = String::new();
        body.push_str(&format!("NAME: {name}\n"));
        body.push_str(TSPLIB_TYPE_TSP_LINE);
        body.push_str(&format!("DIMENSION: {}\n", points.len()));
        body.push_str(TSPLIB_EDGE_WEIGHT_TYPE_LINE);
        body.push_str(TSPLIB_NODE_COORD_SECTION_LINE);

        for (i, p) in points.iter().enumerate() {
            body.push_str(&format!(
                "{} {:.0} {:.0}\n",
                i + TSPLIB_NODE_ID_BASE,
                p.x * EUC2D_SCALE,
                p.y * EUC2D_SCALE
            ));
        }

        body.push_str(TSPLIB_EOF_LINE);
        fs::write(path, body)
    }
}
