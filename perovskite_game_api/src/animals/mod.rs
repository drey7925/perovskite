use crate::default_game::basic_blocks::WATER;
use crate::game_builder::{GameBuilder, StaticTextureName};
use crate::include_texture_bytes;
use cgmath::{vec3, InnerSpace, Vector3, Zero};
use perovskite_core::block_id::BlockId;
use perovskite_core::chat::{ChatMessage, SERVER_ERROR_COLOR};
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_core::protocol;
use perovskite_core::protocol::items::item_def::QuantityType;
use perovskite_core::protocol::render::CustomMesh;
use perovskite_server::game_state;
use perovskite_server::game_state::entities::{
    CoroutineResult, EntityClassId, EntityCoroutine, EntityCoroutineServices, EntityDef,
    EntityMoveDecision, EntityTypeId, MoveQueueType, Movement,
};
use perovskite_server::game_state::event::HandlerContext;
use perovskite_server::game_state::items::ItemStack;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use smallvec::SmallVec;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;

use anyhow::Result;
use perovskite_core::constants::block_groups::DEFAULT_LIQUID;
use perovskite_core::protocol::items::interaction_rule::DigBehavior;
use perovskite_core::protocol::items::{Empty, InteractionRule};

struct DuckCoroutine {
    last_coord: BlockCoordinate,
    last_direction: Vector3<f64>,
    water_block: BlockId,
}

impl Debug for DuckCoroutine {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DuckCoroutine {{..}}")
    }
}

impl EntityCoroutine for DuckCoroutine {
    fn plan_move(
        mut self: Pin<&mut Self>,
        services: &EntityCoroutineServices<'_>,
        _current_position: Vector3<f64>,
        _whence: Vector3<f64>,
        _when: f32,
        _queue_space: usize,
    ) -> CoroutineResult {
        let rng = &mut rand::thread_rng();
        // Stop and wait in one place, probability 0.5

        if rng.gen_bool(0.5) {
            let direction = f64::atan2(self.last_direction.x, self.last_direction.z);
            return CoroutineResult::Successful(EntityMoveDecision::QueueSingleMovement(
                Movement {
                    velocity: Vector3::zero(),
                    acceleration: Vector3::zero(),
                    face_direction: direction as f32,
                    pitch: 0.0,
                    move_time: rng.gen_range(1.0..2.0),
                },
            ));
        }
        // Generate eight candidate moves
        let mut outcomes: SmallVec<[_; 8]> = smallvec::smallvec![];
        let mut weights: SmallVec<[_; 8]> = smallvec::smallvec![];
        for (dx, dy) in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ] {
            if let Some(coord) = self.last_coord.try_delta(dx, 0, dy) {
                if services.try_get_block(coord) == Some(self.water_block) {
                    let direction_vector = Vector3::new(dx as f64, 0.0, dy as f64);
                    let dot = direction_vector
                        .normalize()
                        .dot(self.last_direction.normalize());
                    weights.push(1.25 + dot);
                    outcomes.push(coord);
                }
            }
        }
        if outcomes.is_empty() {
            return CoroutineResult::Successful(EntityMoveDecision::QueueSingleMovement(
                Movement {
                    velocity: Vector3::zero(),
                    acceleration: Vector3::zero(),
                    face_direction: 0.0,
                    pitch: 0.0,
                    move_time: 5.0,
                },
            ));
        }
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen = outcomes[dist.sample(rng)];
        let last_coord_f64 = Vector3::new(
            self.last_coord.x as f64,
            self.last_coord.y as f64,
            self.last_coord.z as f64,
        );
        self.last_direction = Vector3::new(
            chosen.x as f64 - last_coord_f64.x,
            0.0,
            chosen.z as f64 - last_coord_f64.z,
        );
        self.last_coord = chosen;

        CoroutineResult::Successful(EntityMoveDecision::QueueSingleMovement(Movement {
            velocity: self.last_direction.cast().unwrap(),
            acceleration: Vector3::zero(),
            face_direction: f64::atan2(self.last_direction.x, self.last_direction.z) as f32,
            pitch: 0.0,
            move_time: 1.0,
        }))
    }
}

pub fn register_duck(game_builder: &mut GameBuilder) -> Result<()> {
    const DUCK_UV: StaticTextureName = StaticTextureName("animals:duck_uv");
    const DUCK_INV_TEX: StaticTextureName = StaticTextureName("animals:duck_inv");
    include_texture_bytes!(game_builder, DUCK_UV, "textures/duck_uv.png")?;
    include_texture_bytes!(game_builder, DUCK_INV_TEX, "textures/testonly_duck.png")?;

    let id = game_builder.inner.entities_mut().register(EntityDef {
        move_queue_type: MoveQueueType::SingleMove,
        class_name: "animals:duck".to_string(),
        client_info: protocol::entities::EntityAppearance {
            custom_mesh: vec![DUCK_MESH.clone()],
            attachment_offset: Some(vec3(0.0, 0.0, 0.0).try_into()?),
            attachment_offset_in_model_space: false,
        },
    })?;

    game_builder
        .inner
        .items_mut()
        .register_item(game_state::items::Item {
            proto: protocol::items::ItemDef {
                short_name: "animals:duck".to_string(),
                display_name: "Duck".to_string(),
                inventory_texture: Some(DUCK_INV_TEX.into()),
                groups: vec![],
                block_apperance: "".to_string(),
                interaction_rules: vec![InteractionRule {
                    block_group: vec![DEFAULT_LIQUID.to_string()],
                    dig_behavior: Some(DigBehavior::Undiggable(Empty {})),
                    tool_wear: 0,
                }],
                quantity_type: Some(QuantityType::Stack(256)),
                sort_key: "animals:duck".to_string(),
            },
            dig_handler: None,
            tap_handler: None,
            place_handler: Some(Box::new(move |ctx, _placement_coord, anchor, stack| {
                place_duck(ctx, anchor, stack, id)
            })),
        })?;
    Ok(())
}

fn place_duck(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    stack: &ItemStack,
    entity_class: EntityClassId,
) -> Result<Option<ItemStack>> {
    let water_block = ctx
        .block_types()
        .get_by_name(WATER.0)
        .unwrap()
        .with_variant_unchecked(0xfff);
    let block = ctx.game_map().get_block(coord)?;
    if block != water_block {
        ctx.initiator().send_chat_message(
            ChatMessage::new_server_message("Not water source").with_color(SERVER_ERROR_COLOR),
        )?;
        return Ok(Some(stack.clone()));
    }
    let coro = DuckCoroutine {
        last_coord: coord,
        last_direction: Vector3::new(0.0, 0.0, 1.0),
        water_block,
    };
    let id = ctx.entities().new_entity_blocking(
        Vector3::new(coord.x as f64, coord.y as f64 + 0.5, coord.z as f64),
        Some(Box::pin(coro)),
        EntityTypeId {
            class: entity_class,
            data: None,
        },
        None,
    );

    ctx.initiator()
        .send_chat_message(ChatMessage::new_server_message(format!("[{id}]: Quack!")))?;
    Ok(stack.decrement())
}

const DUCK_MESH_BYTES: &[u8] = include_bytes!("duck.obj");
lazy_static::lazy_static! {
    static ref DUCK_MESH: CustomMesh = {
        perovskite_server::formats::load_obj_mesh(DUCK_MESH_BYTES, "animals:duck_uv").unwrap()
    };
}
