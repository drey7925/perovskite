//! Implementation details of light propagation that need to be shared between the client and server
// TODO - this is unused now, because making it generic destroyed the performance
// Get rid of it, or turn it into a slow refernece impl for unit testing
use crate::block_id::BlockId;
#[doc(hidden)]
pub fn do_lighting_pass<F, G, H, I>(
    scratchpad: &mut [u8; 48 * 48 * 48],
    light_emission: F,
    light_propagation: G,
    // Reads an entire row of blocks. TODO consider if we want to batch reads of an entire chunk
    read_row: H,
    read_single: I,
) where
    F: Fn(BlockId) -> u8,
    G: Fn(BlockId) -> bool,
    // (i / 16, j, k), e.g. (-1, -5, 21)
    H: Fn(i32, i32, i32) -> [BlockId; 16],
    I: Fn(i32, i32, i32) -> BlockId,
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
    
    scratchpad.fill(0);

    let mut queue = vec![];
    // First, scan through the neighborhood looking for light sources
    // Indices are reversed in order to achieve better cache locality
    // i is the minor index, j is intermediate, and k is the major index
    let mut light_levels = [0; 48];
    for k in -16i32..32 {
        for j in -16i32..32 {
            for i in -1..=1 {
                let row = read_row(i, j, k);
                for x in 0..16 {
                    light_levels[((i + 1) * 16) as usize + x] = light_emission(row[x]);
                }
            }
            for i in 0..48 {
                if light_levels[i as usize] > 0 {
                    // We have some light. Check if it could possibly reach our own block
                    maybe_push(&mut queue, i - 16, j, k, light_levels[i as usize]);
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

        let light_propagate_for_coord = |i, j, k| light_propagation(read_single(i, j, k));

        check_propagation_and_push(
            &mut queue,
            i - 1,
            j,
            k,
            light_level - 1,
            &light_propagate_for_coord,
        );
        check_propagation_and_push(
            &mut queue,
            i + 1,
            j,
            k,
            light_level - 1,
            &light_propagate_for_coord,
        );
        check_propagation_and_push(
            &mut queue,
            i,
            j - 1,
            k,
            light_level - 1,
            &light_propagate_for_coord,
        );
        check_propagation_and_push(
            &mut queue,
            i,
            j + 1,
            k,
            light_level - 1,
            &light_propagate_for_coord,
        );
        check_propagation_and_push(
            &mut queue,
            i,
            j,
            k - 1,
            light_level - 1,
            &light_propagate_for_coord,
        );
        check_propagation_and_push(
            &mut queue,
            i,
            j,
            k + 1,
            light_level - 1,
            &light_propagate_for_coord,
        );
    }
}
