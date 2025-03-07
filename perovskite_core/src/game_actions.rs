use crate::block_id::BlockId;
use crate::coordinates::BlockCoordinate;
use crate::protocol::game_rpc::dig_tap_action::ActionTarget;
use crate::protocol::game_rpc::interact_key_action::InteractionTarget;
use crate::protocol::game_rpc::place_action::PlaceAnchor;
use crate::protocol::game_rpc::EntityTarget;

/// The thing which is being dug/tapped or was dug/tapped
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ToolTarget {
    /// Block coordinate, and the ID at that coordinate at the time the target was calculated
    Block(BlockCoordinate),
    /// Selected entity (entity id, trailing entity index), entity class
    Entity(EntityTarget),
}

impl From<ToolTarget> for InteractionTarget {
    fn from(value: ToolTarget) -> Self {
        match value {
            ToolTarget::Block(coord) => InteractionTarget::BlockCoord(coord.into()),
            ToolTarget::Entity(target) => InteractionTarget::EntityTarget(target),
        }
    }
}

impl From<ToolTarget> for ActionTarget {
    fn from(value: ToolTarget) -> Self {
        match value {
            ToolTarget::Block(coord) => ActionTarget::BlockCoord(coord.into()),
            ToolTarget::Entity(target) => ActionTarget::EntityTarget(target),
        }
    }
}

impl From<ToolTarget> for PlaceAnchor {
    fn from(value: ToolTarget) -> Self {
        match value {
            ToolTarget::Block(coord) => PlaceAnchor::AnchorBlock(coord.into()),
            ToolTarget::Entity(target) => PlaceAnchor::AnchorEntity(target),
        }
    }
}

impl From<PlaceAnchor> for ToolTarget {
    fn from(value: PlaceAnchor) -> Self {
        match value {
            PlaceAnchor::AnchorBlock(coord) => ToolTarget::Block(coord.into()),
            PlaceAnchor::AnchorEntity(target) => ToolTarget::Entity(target),
        }
    }
}
