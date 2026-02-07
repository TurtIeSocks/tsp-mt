use crate::utils::{self, Point};

use super::{
    merge::merge_cycles,
    refine::refine_cycle_seams,
    split::{
        add_halos_binary, add_halos_quadtree, best_order_by_centroids, split_long_axis,
        split_quadtree_median,
    },
    Options,
};

pub(crate) fn solve_rec(points: Vec<Point>, opts: &Options) -> std::io::Result<Vec<Point>> {
    let n = points.len();

    // Leaf: call external tsp.
    // Note: halos can inflate n. That's OK. The strict runner ensures tsp returns expected count.
    if n <= opts.max_leaf_size {
        return utils::run_external_tsp_strict(&opts.tsp_path, &points);
    }

    let use_quadtree = n >= 4 * opts.leaf_size;

    if use_quadtree {
        let (children, med_lat, med_lng) = split_quadtree_median(points);
        let children = add_halos_quadtree(children, med_lat, med_lng, opts.halo);

        // MOVE children into tasks (no clones).
        let [c0, c1, c2, c3] = children;

        let (r0, r1) = rayon::join(
            || {
                if c0.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c0, opts)
                }
            },
            || {
                if c1.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c1, opts)
                }
            },
        );
        let (r2, r3) = rayon::join(
            || {
                if c2.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c2, opts)
                }
            },
            || {
                if c3.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c3, opts)
                }
            },
        );

        let solved: [Vec<Point>; 4] = [r0?, r1?, r2?, r3?];

        // Order children by centroid mini-tour (size <= 4) and merge sequentially.
        let order = best_order_by_centroids(&solved);
        let mut merged: Vec<Point> = Vec::new();

        for &idx in &order {
            let child_tour = &solved[idx];
            if child_tour.is_empty() {
                continue;
            }
            merged = if merged.is_empty() {
                child_tour.clone()
            } else {
                merge_cycles(&merged, child_tour, opts.portals)
            };
            if opts.seam_refine {
                refine_cycle_seams(&mut merged, opts.seam_window);
            }
        }

        return Ok(merged);
    }

    // Binary split fallback (still supports halos).
    let (left, right, boundary, split_axis_lat) = split_long_axis(points);
    let (left, right) = add_halos_binary(left, right, boundary, split_axis_lat, opts.halo);

    let (a_res, b_res) = rayon::join(|| solve_rec(left, opts), || solve_rec(right, opts));
    let a = a_res?;
    let b = b_res?;

    let mut merged = merge_cycles(&a, &b, opts.portals);
    if opts.seam_refine {
        refine_cycle_seams(&mut merged, opts.seam_window);
    }
    Ok(merged)
}
