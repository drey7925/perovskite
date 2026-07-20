use crate::{
    blocks::BlockBuilder,
    carts::network::{AdjacencyHit, CachedHit},
    game_builder::{GameBuilder, OwnedTextureName},
};
use anyhow::Result;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_server::game_state::event::HandlerContext;

#[derive(Clone, Debug)]
pub(crate) struct StationRoute {
    pub(super) dest: Option<AdjacencyHit>,
    pub(super) via: Vec<AdjacencyHit>,
    #[allow(unused)]
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

#[derive(Clone, prost::Message)]
pub(crate) struct TempTestonlyHardcodedStationConfig {
    #[prost(message, tag = "1")]
    pub(super) dest: Option<CachedHit>,
    #[prost(message, repeated, tag = "2")]
    pub(super) via: Vec<CachedHit>,
    #[prost(message, tag = "3")]
    pub(super) controller_coord: Option<BlockCoordinate>,
}
impl TryFrom<TempTestonlyHardcodedStationConfig> for StationRoute {
    type Error = anyhow::Error;
    fn try_from(value: TempTestonlyHardcodedStationConfig) -> Result<Self, Self::Error> {
        let dest = value.dest.map(|x| x.try_into()).transpose()?;
        let via = value
            .via
            .into_iter()
            .map(|x| x.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        let controller_coord = value
            .controller_coord
            .ok_or_else(|| anyhow::anyhow!("controller_coord not set"))?;
        Ok(StationRoute {
            dest,
            via,
            controller_coord,
        })
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

    let (_bt, maybe_data) = ctx.game_map().get_block_with_extended_data(
        controller_coord,
        |_, ext| -> Result<Option<TempTestonlyHardcodedStationConfig>> {
            Ok(ext
                .custom_data_ref::<TempTestonlyHardcodedStationConfig>()
                .cloned())
        },
    )?;

    match maybe_data {
        Some(data) => {
            dbg!(data.try_into())
        }
        None => Ok(StationRoute {
            dest: None,
            via: Vec::new(),
            controller_coord,
        }),
    }
}

pub fn register_station_controller(builder: &mut GameBuilder) -> Result<()> {
    BlockBuilder::new("carts:station_controller")
        .set_display_name("TAMA_neko") // Wakayama Dentetsu, where else?
        .set_cube_single_texture(OwnedTextureName::from_css_color("orange"))
        .add_modifier(|bt| {
            bt.register_proto_serialization_handlers::<TempTestonlyHardcodedStationConfig>()
        })
        .build_and_deploy_into(builder)?;
    Ok(())
}
