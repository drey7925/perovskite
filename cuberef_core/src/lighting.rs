//! Implementation details of light propagation that need to be shared between the client and server
#[doc(hidden)]
pub fn do_lighting_pass<F, G>(
    scratchpad: &mut [u8; 48 * 48 * 48],
    light_emission: F,
    light_propagation: G,
) where
    F: Fn(i32, i32, i32) -> u8,
    G: Fn(i32, i32, i32) -> bool,
{
    #[inline]
    fn maybe_push(queue: &mut Vec<(i32, i32, i32, u8)>, i: i32, j: i32, k: i32, light_level: u8) {
        if i < -16 || j < -16 || k < -16 || i >= 32 || j >= 32 || k >= 32 {
            return;
        }
        let i_dist = (-1 - i).max(i - 16);
        let j_dist = (-1 - j).max(j - 16);
        let k_dist = (-1 - k).max(k - 16);
        let dist = i_dist + j_dist + k_dist;
        if dist < (light_level as i32) {
            queue.push((i, j, k, light_level));
        }
    }

    #[inline]
    fn check_propagation_and_push<F>(
        queue: &mut Vec<(i32, i32, i32, u8)>,
        i: i32,
        j: i32,
        k: i32,
        light_level: u8,
        light_propagation: F,
    ) where
        F: Fn(i32, i32, i32) -> bool,
    {
        if i < -16 || j < -16 || k < -16 || i >= 32 || j >= 32 || k >= 32 {
            return;
        }
        if !light_propagation(i, j, k) {
            return;
        }
        let i_dist = (-1 - i).max(i - 16);
        let j_dist = (-1 - j).max(j - 16);
        let k_dist = (-1 - k).max(k - 16);
        let dist = i_dist + j_dist + k_dist;
        if dist < (light_level as i32) {
            queue.push((i, j, k, light_level));
        }
    }

    let mut queue = vec![];
    // First, scan through the neighborhood looking for light sources
    for i in -16i32..32 {
        for j in -16i32..32 {
            for k in -16i32..32 {
                // Check if we have lighting for this block
                let light_level = light_emission(i, j, k);
                if light_level > 0 {
                    // We have some light. Check if it could possibly reach our own block
                    maybe_push(&mut queue, i, j, k, light_level);
                }
            }
        }
    }
    // Then, while the queue is non-empty, attempt to propagate light
    while !queue.is_empty() {
        let (i, j, k, light_level) = queue.pop().unwrap();
        let old_level =
            scratchpad[(i + 16) as usize * 48 * 48 + (j + 16) as usize * 48 + (k + 16) as usize];
        if old_level >= light_level {
            continue;
        }
        // Set the queued light value
        scratchpad[(i + 16) as usize * 48 * 48 + (j + 16) as usize * 48 + (k + 16) as usize] =
            light_level;

        check_propagation_and_push(&mut queue, i - 1, j, k, light_level - 1, &light_propagation);
        check_propagation_and_push(&mut queue, i + 1, j, k, light_level - 1, &light_propagation);
        check_propagation_and_push(&mut queue, i, j - 1, k, light_level - 1, &light_propagation);
        check_propagation_and_push(&mut queue, i, j + 1, k, light_level - 1, &light_propagation);
        check_propagation_and_push(&mut queue, i, j, k - 1, light_level - 1, &light_propagation);
        check_propagation_and_push(&mut queue, i, j, k + 1, light_level - 1, &light_propagation);
    }
}
