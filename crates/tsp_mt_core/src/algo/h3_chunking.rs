use std::collections::HashMap;

use h3o::{CellIndex, LatLng, Resolution};

use crate::{Error, Result, node::LKHNode};

const INITIAL_RESOLUTION: Resolution = Resolution::Four;
const MAX_RESOLUTION: Resolution = Resolution::Fifteen;
const ERR_INVALID_LAT_LNG: &str = "invalid lat/lng for H3";

#[derive(Clone, Debug)]
pub(crate) struct ChunkPartition {
    pub(crate) indices: Vec<usize>,
    pub(crate) cell: CellIndex,
    pub(crate) resolution: Resolution,
}

#[derive(Clone, Debug)]
struct ChunkBucket {
    indices: Vec<usize>,
    cell: CellIndex,
    resolution: Resolution,
}

pub(crate) fn partition_with_metadata(
    input: &[LKHNode],
    max_bucket_sz: usize,
) -> Result<Vec<ChunkPartition>> {
    let mut res = INITIAL_RESOLUTION;
    let mut buckets = bucket_by_res(input, res, None)?;

    while max_bucket(&buckets) > max_bucket_sz && res != MAX_RESOLUTION {
        res = res.succ().unwrap_or(Resolution::Zero);
        buckets = bucket_by_res(input, res, None)?;
    }

    let mut out: Vec<ChunkPartition> = Vec::new();

    for (cell, idxs) in buckets {
        if idxs.len() <= max_bucket_sz {
            out.push(ChunkPartition {
                indices: idxs,
                cell,
                resolution: res,
            });
            continue;
        }

        let mut local_res = res;
        let mut frontier: Vec<ChunkBucket> = vec![ChunkBucket {
            indices: idxs,
            cell,
            resolution: local_res,
        }];

        while local_res != MAX_RESOLUTION
            && frontier
                .iter()
                .any(|bucket| bucket.indices.len() > max_bucket_sz)
        {
            local_res = local_res.succ().unwrap_or(Resolution::Zero);
            let mut next_frontier: Vec<ChunkBucket> = Vec::new();

            for bucket in frontier {
                if bucket.indices.len() <= max_bucket_sz {
                    next_frontier.push(bucket);
                    continue;
                }
                let sub = bucket_by_res(input, local_res, Some(&bucket.indices))?;
                for (sub_cell, sub_indices) in sub {
                    next_frontier.push(ChunkBucket {
                        indices: sub_indices,
                        cell: sub_cell,
                        resolution: local_res,
                    });
                }
            }
            frontier = next_frontier;
        }

        for mut bucket in frontier {
            if bucket.indices.len() > max_bucket_sz {
                bucket.indices.sort_unstable();
                for chunk_indices in bucket.indices.chunks(max_bucket_sz) {
                    out.push(ChunkPartition {
                        indices: chunk_indices.to_vec(),
                        cell: bucket.cell,
                        resolution: bucket.resolution,
                    });
                }
            } else {
                out.push(ChunkPartition {
                    indices: bucket.indices,
                    cell: bucket.cell,
                    resolution: bucket.resolution,
                });
            }
        }
    }

    out.sort_unstable_by(|lhs, rhs| {
        u8::from(lhs.resolution)
            .cmp(&u8::from(rhs.resolution))
            .then_with(|| u64::from(lhs.cell).cmp(&u64::from(rhs.cell)))
            .then_with(|| {
                lhs.indices
                    .first()
                    .copied()
                    .unwrap_or(usize::MAX)
                    .cmp(&rhs.indices.first().copied().unwrap_or(usize::MAX))
            })
            .then_with(|| lhs.indices.len().cmp(&rhs.indices.len()))
    });

    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn partition_indices(
    input: &[LKHNode],
    max_bucket_sz: usize,
) -> Result<Vec<Vec<usize>>> {
    partition_with_metadata(input, max_bucket_sz)
        .map(|chunks| chunks.into_iter().map(|chunk| chunk.indices).collect())
}

fn bucket_by_res(
    input: &[LKHNode],
    res: Resolution,
    subset: Option<&[usize]>,
) -> Result<HashMap<CellIndex, Vec<usize>>> {
    let mut map: HashMap<CellIndex, Vec<usize>> = HashMap::new();

    let mut add_index = |i: usize| -> Result<()> {
        let p = &input[i];
        let ll =
            LatLng::new(p.lat(), p.lng()).map_err(|_| Error::invalid_input(ERR_INVALID_LAT_LNG))?;
        let cell = ll.to_cell(res);
        map.entry(cell).or_default().push(i);
        Ok(())
    };

    if let Some(idxs) = subset {
        for &i in idxs {
            add_index(i)?;
        }
    } else {
        for i in 0..input.len() {
            add_index(i)?;
        }
    }

    Ok(map)
}

fn max_bucket(map: &HashMap<CellIndex, Vec<usize>>) -> usize {
    map.values().map(|v| v.len()).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::LKHNode;

    #[test]
    fn partition_indices_respects_max_bucket_size_and_preserves_all_indices() {
        let input = vec![
            LKHNode::from_lat_lng(37.7749, -122.4194),
            LKHNode::from_lat_lng(37.7750, -122.4195),
            LKHNode::from_lat_lng(37.7751, -122.4196),
            LKHNode::from_lat_lng(34.0522, -118.2437),
            LKHNode::from_lat_lng(40.7128, -74.0060),
        ];

        let chunks = partition_indices(&input, 2).expect("partition should succeed");
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.len() <= 2));

        let mut flattened: Vec<usize> = chunks.into_iter().flatten().collect();
        flattened.sort_unstable();
        assert_eq!(flattened, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn partition_with_metadata_tracks_resolution_and_preserves_all_indices() {
        let input = vec![
            LKHNode::from_lat_lng(37.7749, -122.4194),
            LKHNode::from_lat_lng(37.7750, -122.4195),
            LKHNode::from_lat_lng(37.7751, -122.4196),
            LKHNode::from_lat_lng(34.0522, -118.2437),
            LKHNode::from_lat_lng(40.7128, -74.0060),
        ];

        let chunks = partition_with_metadata(&input, 2).expect("partition should succeed");
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.indices.len() <= 2));
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.cell.resolution() == chunk.resolution)
        );

        let mut flattened: Vec<usize> =
            chunks.into_iter().flat_map(|chunk| chunk.indices).collect();
        flattened.sort_unstable();
        assert_eq!(flattened, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn partition_with_metadata_is_deterministic_for_same_input() {
        let input = vec![
            LKHNode::from_lat_lng(37.7749, -122.4194),
            LKHNode::from_lat_lng(37.7750, -122.4195),
            LKHNode::from_lat_lng(37.7751, -122.4196),
            LKHNode::from_lat_lng(34.0522, -118.2437),
            LKHNode::from_lat_lng(40.7128, -74.0060),
            LKHNode::from_lat_lng(47.6062, -122.3321),
            LKHNode::from_lat_lng(25.7617, -80.1918),
        ];

        let a = partition_with_metadata(&input, 2).expect("partition should succeed");
        let b = partition_with_metadata(&input, 2).expect("partition should succeed");

        let key = |chunks: Vec<ChunkPartition>| -> Vec<(u8, u64, Vec<usize>)> {
            chunks
                .into_iter()
                .map(|chunk| {
                    (
                        u8::from(chunk.resolution),
                        u64::from(chunk.cell),
                        chunk.indices,
                    )
                })
                .collect()
        };

        assert_eq!(key(a), key(b));
    }

    #[test]
    fn partition_indices_rejects_invalid_lat_lng() {
        let input = vec![LKHNode::from_lat_lng(f64::NAN, 0.0)];
        let err = partition_indices(&input, 1).expect_err("expected invalid input");
        assert!(err.to_string().contains("invalid lat/lng for H3"));
    }
}
