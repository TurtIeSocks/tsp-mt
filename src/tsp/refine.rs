use crate::utils::Point;

/// A small seam improvement for cycles.
/// Implementation: rotate to a few offsets, run the existing open seam refinement, rotate back.
pub(crate) fn refine_cycle_seams(route: &mut Vec<Point>, window: usize) {
    let n = route.len();
    if n < 8 {
        return;
    }
    let offsets = [0usize, n / 3, (2 * n) / 3];

    for &off in &offsets {
        route.rotate_left(off);
        refine_seams_small_open(route, window);
        route.rotate_right(off);
    }
}

/// Your original seam refinement logic (open). Kept as a helper.
fn refine_seams_small_open(route: &mut [Point], window: usize) {
    let n = route.len();
    if n < 6 {
        return;
    }
    let w = window.min(n / 2).max(8);

    for i in 1..(n - 3) {
        let a = route[i - 1];
        let b = route[i];
        let j_max = (i + w).min(n - 2);

        for j in (i + 1)..=j_max {
            let c = route[j];
            let d = route[j + 1];
            let before = a.dist(&b) + c.dist(&d);
            let after = a.dist(&c) + b.dist(&d);
            if after + 1e-9 < before {
                route[i..=j].reverse();
            }
        }
    }
}
