use crate::carts::network::AdjacencyHit;
use anyhow::Result;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_server::game_state::event::HandlerContext;

#[derive(Clone, Debug)]
pub(crate) struct StationRoute {
    pub(super) dest: Option<AdjacencyHit>,
    pub(super) via: Vec<AdjacencyHit>,
    pub(super) controller_coord: BlockCoordinate,
}
impl StationRoute {
    pub(super) fn next_hop(&self) -> Option<&AdjacencyHit> {
        if let Some(via) = self.via.first() {
            Some(via)
        } else if let Some(dest) = self.dest.as_ref() {
            Some(dest)
        } else {
            None
        }
    }

    pub(super) fn drop_next_hop(&mut self) {
        if self.via.first().is_some() {
            self.via.remove(0);
        } else {
            self.dest = None;
        }
    }
}

/// Returns a route for the given cart through the station whose controller is at
/// `controller_coord`. `cart_name` is ignored for now, and may be extended to take other cart details.
///
/// A "route" is a sequence of `AdjacencyHit` objects representing the segments the cart should
/// take to travel through the station to its destination.
///
/// This should NOT include the very first signal where the cart enters, as part of the route - the cart
/// is already there.
pub fn request_routing(
    ctx: &HandlerContext<'_>,
    controller_coord: BlockCoordinate,
    _cart_name: &str,
) -> Result<StationRoute> {
    // TODO: Implement actual routing
    Ok(StationRoute {
        dest: None,
        via: Vec::new(),
        controller_coord,
    })
}
