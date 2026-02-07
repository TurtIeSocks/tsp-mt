use std::{
    cmp::Ordering,
    collections::HashSet,
    hash::{Hash, Hasher},
};

use rayon::slice::ParallelSliceMut;

use crate::utils::Point;

#[derive(Clone, Copy, Debug, Eq)]
struct Key(u64, u64);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0);
        state.write_u64(self.1);
    }
}
fn key(p: Point) -> Key {
    Key(p.lat.to_bits(), p.lng.to_bits())
}

/// Quadtree by median lat and median lng.
/// Returns: children[4], med_lat, med_lng
pub(crate) fn split_quadtree_median(pts: Vec<Point>) -> ([Vec<Point>; 4], f64, f64) {
    let n = pts.len();
    let mut lats: Vec<f64> = pts.iter().map(|p| p.lat).collect();
    let mut lngs: Vec<f64> = pts.iter().map(|p| p.lng).collect();

    lats.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    lngs.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let med_lat = lats[n / 2];
    let med_lng = lngs[n / 2];

    let mut q0 = Vec::new();
    let mut q1 = Vec::new();
    let mut q2 = Vec::new();
    let mut q3 = Vec::new();

    for p in pts {
        let lat_hi = p.lat >= med_lat;
        let lng_hi = p.lng >= med_lng;
        match (lat_hi, lng_hi) {
            (false, false) => q0.push(p),
            (false, true) => q1.push(p),
            (true, false) => q2.push(p),
            (true, true) => q3.push(p),
        }
    }

    ([q0, q1, q2, q3], med_lat, med_lng)
}

/// Add halos near the median boundaries in a quadtree split.
pub(crate) fn add_halos_quadtree(
    mut children: [Vec<Point>; 4],
    med_lat: f64,
    med_lng: f64,
    halo: usize,
) -> [Vec<Point>; 4] {
    if halo == 0 {
        return children;
    }

    let mut bucket_keys: [HashSet<Key>; 4] = [
        children[0].iter().copied().map(key).collect(),
        children[1].iter().copied().map(key).collect(),
        children[2].iter().copied().map(key).collect(),
        children[3].iter().copied().map(key).collect(),
    ];

    // Collect all points once.
    let mut all: Vec<Point> = Vec::new();
    for c in &children {
        all.extend_from_slice(c);
    }

    all.par_sort_unstable_by(|a, b| {
        a.lat
            .partial_cmp(&b.lat)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.lng.partial_cmp(&b.lng).unwrap_or(Ordering::Equal))
    });
    all.dedup_by(|a, b| key(*a) == key(*b));

    // Near lat boundary:
    let mut near_lat: Vec<(f64, Point)> = all
        .iter()
        .copied()
        .map(|p| ((p.lat - med_lat).abs(), p))
        .collect();
    near_lat.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    for &(_, p) in near_lat.iter().take(halo) {
        let lat_hi = p.lat >= med_lat;
        let lng_hi = p.lng >= med_lng;
        let dst = bucket_index(!lat_hi, lng_hi);
        let k = key(p);
        if !bucket_keys[dst].contains(&k) {
            children[dst].push(p);
            bucket_keys[dst].insert(k);
        }
    }

    // Near lng boundary:
    let mut near_lng: Vec<(f64, Point)> = all
        .iter()
        .copied()
        .map(|p| ((p.lng - med_lng).abs(), p))
        .collect();
    near_lng.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    for &(_, p) in near_lng.iter().take(halo) {
        let lat_hi = p.lat >= med_lat;
        let lng_hi = p.lng >= med_lng;
        let dst = bucket_index(lat_hi, !lng_hi);
        let k = key(p);
        if !bucket_keys[dst].contains(&k) {
            children[dst].push(p);
            bucket_keys[dst].insert(k);
        }
    }

    children
}

fn bucket_index(lat_hi: bool, lng_hi: bool) -> usize {
    match (lat_hi, lng_hi) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    }
}

/// Binary split by longest axis (lat or lng).
/// Returns (left, right, boundary_value, split_axis_is_lat).
pub(crate) fn split_long_axis(mut pts: Vec<Point>) -> (Vec<Point>, Vec<Point>, f64, bool) {
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    let mut min_lng = f64::INFINITY;
    let mut max_lng = f64::NEG_INFINITY;

    for p in &pts {
        min_lat = min_lat.min(p.lat);
        max_lat = max_lat.max(p.lat);
        min_lng = min_lng.min(p.lng);
        max_lng = max_lng.max(p.lng);
    }

    let lat_span = max_lat - min_lat;
    let lng_span = max_lng - min_lng;
    let split_axis_is_lat = lat_span >= lng_span;

    if split_axis_is_lat {
        pts.par_sort_unstable_by(|a, b| a.lat.partial_cmp(&b.lat).unwrap_or(Ordering::Equal));
    } else {
        pts.par_sort_unstable_by(|a, b| a.lng.partial_cmp(&b.lng).unwrap_or(Ordering::Equal));
    }

    let mid = pts.len() / 2;
    let boundary = if split_axis_is_lat {
        pts[mid].lat
    } else {
        pts[mid].lng
    };
    let right = pts.split_off(mid);
    (pts, right, boundary, split_axis_is_lat)
}

/// Add halos near a binary boundary.
pub(crate) fn add_halos_binary(
    mut left: Vec<Point>,
    mut right: Vec<Point>,
    boundary: f64,
    axis_lat: bool,
    halo: usize,
) -> (Vec<Point>, Vec<Point>) {
    if halo == 0 {
        return (left, right);
    }

    let mut left_keys: HashSet<Key> = left.iter().copied().map(key).collect();
    let mut right_keys: HashSet<Key> = right.iter().copied().map(key).collect();

    let mut candidates: Vec<(f64, Point)> = left
        .iter()
        .chain(right.iter())
        .copied()
        .map(|p| {
            let d = if axis_lat {
                (p.lat - boundary).abs()
            } else {
                (p.lng - boundary).abs()
            };
            (d, p)
        })
        .collect();

    candidates.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    for &(_, p) in candidates.iter().take(halo) {
        let k = key(p);
        if !left_keys.contains(&k) {
            left.push(p);
            left_keys.insert(k);
        }
        if !right_keys.contains(&k) {
            right.push(p);
            right_keys.insert(k);
        }
    }

    (left, right)
}

fn centroid(points: &[Point]) -> Point {
    let mut s_lat = 0.0;
    let mut s_lng = 0.0;
    for p in points {
        s_lat += p.lat;
        s_lng += p.lng;
    }
    let n = points.len() as f64;
    Point {
        lat: s_lat / n,
        lng: s_lng / n,
    }
}

/// Returns an order of indices for non-empty chunks by brute-forcing permutations (<= 4! = 24).
pub(crate) fn best_order_by_centroids(chunks: &[Vec<Point>; 4]) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..4).filter(|&i| !chunks[i].is_empty()).collect();
    if idxs.len() <= 1 {
        return idxs;
    }

    let cents: [Point; 4] = [
        if chunks[0].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            centroid(&chunks[0])
        },
        if chunks[1].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            centroid(&chunks[1])
        },
        if chunks[2].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            centroid(&chunks[2])
        },
        if chunks[3].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            centroid(&chunks[3])
        },
    ];

    let mut best = idxs.clone();
    let mut best_cost = f64::INFINITY;

    permute(&mut idxs, 0, &mut |perm| {
        let mut cost = 0.0;
        for w in perm.windows(2) {
            cost += cents[w[0]].dist(&cents[w[1]]);
        }
        if cost < best_cost {
            best_cost = cost;
            best = perm.to_vec();
        }
    });

    best
}

fn permute<F: FnMut(&[usize])>(a: &mut [usize], k: usize, f: &mut F) {
    if k == a.len() {
        f(a);
        return;
    }
    for i in k..a.len() {
        a.swap(k, i);
        permute(a, k + 1, f);
        a.swap(k, i);
    }
}
