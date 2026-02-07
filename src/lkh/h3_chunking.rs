use std::collections::HashMap;

use h3o::{CellIndex, LatLng, Resolution};

use crate::utils::Point;

pub(crate) struct H3Chunker;

impl H3Chunker {
    pub(crate) fn partition_indices(input: &[Point], max_bucket_sz: usize) -> Vec<Vec<usize>> {
        const MAX_RES: Resolution = Resolution::Fifteen;

        let mut res = Resolution::Four;
        let mut buckets = Self::bucket_by_res(input, res, None);

        while Self::max_bucket(&buckets) > max_bucket_sz && res != MAX_RES {
            res = Self::next_resolution(res);
            buckets = Self::bucket_by_res(input, res, None);
        }

        let mut out: Vec<Vec<usize>> = Vec::new();

        for (_cell, idxs) in buckets {
            if idxs.len() <= max_bucket_sz {
                out.push(idxs);
                continue;
            }

            let mut local_res = res;
            let mut frontier: Vec<Vec<usize>> = vec![idxs];

            while local_res != MAX_RES && frontier.iter().any(|b| b.len() > max_bucket_sz) {
                local_res = Self::next_resolution(local_res);
                let mut next_frontier: Vec<Vec<usize>> = Vec::new();

                for b in frontier {
                    if b.len() <= max_bucket_sz {
                        next_frontier.push(b);
                        continue;
                    }
                    let sub = Self::bucket_by_res(input, local_res, Some(&b));
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

        out
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
        input: &[Point],
        res: Resolution,
        subset: Option<&[usize]>,
    ) -> HashMap<CellIndex, Vec<usize>> {
        let mut map: HashMap<CellIndex, Vec<usize>> = HashMap::new();
        let iter: Box<dyn Iterator<Item = usize>> = match subset {
            Some(idxs) => Box::new(idxs.iter().copied()),
            None => Box::new((0..input.len()).into_iter()),
        };

        for i in iter {
            let p = &input[i];
            let ll = LatLng::new(p.lat, p.lng).expect("valid lat/lng");
            let cell = ll.to_cell(res);
            map.entry(cell).or_default().push(i);
        }

        map
    }

    fn max_bucket(map: &HashMap<CellIndex, Vec<usize>>) -> usize {
        map.values().map(|v| v.len()).max().unwrap_or(0)
    }
}
