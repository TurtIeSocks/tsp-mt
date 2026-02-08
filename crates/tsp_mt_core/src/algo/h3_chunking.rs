use std::collections::HashMap;

use h3o::{CellIndex, LatLng, Resolution};

use crate::{Error, Result, node::LKHNode};

const INITIAL_RESOLUTION: Resolution = Resolution::Four;
const MAX_RESOLUTION: Resolution = Resolution::Fifteen;
const ERR_INVALID_LAT_LNG: &str = "invalid lat/lng for H3";

pub(crate) struct H3Chunker;

impl H3Chunker {
    pub(crate) fn partition_indices(
        input: &[LKHNode],
        max_bucket_sz: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let mut res = INITIAL_RESOLUTION;
        let mut buckets = Self::bucket_by_res(input, res, None)?;

        while Self::max_bucket(&buckets) > max_bucket_sz && res != MAX_RESOLUTION {
            res = Self::next_resolution(res);
            buckets = Self::bucket_by_res(input, res, None)?;
        }

        let mut out: Vec<Vec<usize>> = Vec::new();

        for (_cell, idxs) in buckets {
            if idxs.len() <= max_bucket_sz {
                out.push(idxs);
                continue;
            }

            let mut local_res = res;
            let mut frontier: Vec<Vec<usize>> = vec![idxs];

            while local_res != MAX_RESOLUTION && frontier.iter().any(|b| b.len() > max_bucket_sz) {
                local_res = Self::next_resolution(local_res);
                let mut next_frontier: Vec<Vec<usize>> = Vec::new();

                for b in frontier {
                    if b.len() <= max_bucket_sz {
                        next_frontier.push(b);
                        continue;
                    }
                    let sub = Self::bucket_by_res(input, local_res, Some(&b))?;
                    for v in sub.into_values() {
                        next_frontier.push(v);
                    }
                }
                frontier = next_frontier;
            }

            for mut b in frontier {
                if b.len() > max_bucket_sz {
                    b.sort_unstable();
                    for c in b.chunks(max_bucket_sz) {
                        out.push(c.to_vec());
                    }
                } else {
                    out.push(b);
                }
            }
        }

        Ok(out)
    }

    fn next_resolution(r: Resolution) -> Resolution {
        use Resolution::*;
        match r {
            Zero => One,
            One => Two,
            Two => Three,
            Three => Four,
            Four => Five,
            Five => Six,
            Six => Seven,
            Seven => Eight,
            Eight => Nine,
            Nine => Ten,
            Ten => Eleven,
            Eleven => Twelve,
            Twelve => Thirteen,
            Thirteen => Fourteen,
            Fourteen => Fifteen,
            Fifteen => Fifteen,
        }
    }

    fn bucket_by_res(
        input: &[LKHNode],
        res: Resolution,
        subset: Option<&[usize]>,
    ) -> Result<HashMap<CellIndex, Vec<usize>>> {
        let mut map: HashMap<CellIndex, Vec<usize>> = HashMap::new();

        let mut add_index = |i: usize| -> Result<()> {
            let p = &input[i];
            let ll = LatLng::new(p.lat(), p.lng())
                .map_err(|_| Error::invalid_input(ERR_INVALID_LAT_LNG))?;
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
}

#[cfg(test)]
mod tests {
    use super::H3Chunker;
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

        let chunks = H3Chunker::partition_indices(&input, 2).expect("partition should succeed");
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.len() <= 2));

        let mut flattened: Vec<usize> = chunks.into_iter().flatten().collect();
        flattened.sort_unstable();
        assert_eq!(flattened, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn partition_indices_rejects_invalid_lat_lng() {
        let input = vec![LKHNode::from_lat_lng(f64::NAN, 0.0)];
        let err = H3Chunker::partition_indices(&input, 1).expect_err("expected invalid input");
        assert!(err.to_string().contains("invalid lat/lng for H3"));
    }
}
