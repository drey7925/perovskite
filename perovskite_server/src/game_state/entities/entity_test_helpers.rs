// use cgmath::Vector3;
// use std::{pin::Pin, sync::Arc};

// use crate::game_state::{
//     entities::{EntityCoroutine, EntityCoroutineServices, MoveQueue},
//     GameState,
// };
// WIP
// pub struct CoroutineTester {
//     coro: Pin<Box<dyn EntityCoroutine>>,
//     move_queue: MoveQueue,
//     last_tick: u64,
//     last_pos: Vector3<f64>,
// }
// impl CoroutineTester {
//     pub fn advance(&mut self, gs: Arc<GameState>) {
//         let services = EntityCoroutineServices { game_state: &gs };
//         self.coro.as_mut().plan_move(
//             services,
//             current_position,
//             whence,
//             when,
//             self.move_queue.remaining_capacity(),
//         );
//     }
// }
