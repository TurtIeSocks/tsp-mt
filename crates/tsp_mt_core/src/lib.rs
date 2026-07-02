//! Multi-threaded TSP solving on geographic coordinates using a built-in
//! clean-room solver (candidate-list 2-opt/Or-opt local search with
//! parallel segment-based iterated local search).

mod error;
mod io;
pub mod logging;
mod node;
pub mod solver;
mod tour;

pub(crate) use io::options;

pub use error::{Error, Result};
pub use io::input::SolverInput;
pub use io::options::SolverOptions;
pub use node::GeoPoint;
pub use tour::Tour;
