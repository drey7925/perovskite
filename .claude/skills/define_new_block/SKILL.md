---
name: define_new_block
description: Defines a new block using BlockBuilder. Use this when asked to add a new block type to the game. Also provides guidance on textures.
---

The main block API is in `perovskite_game_api/src/blocks.rs`. Blocks are defined using `BlockBuilder` and registered via `game_builder.add_block(...)`.

## Basic Structure

```rust
let built_block = builder.add_block(
    BlockBuilder::new(BLOCK_NAME)
        // ... configuration ...
        .set_cube_single_texture(SOME_TEXTURE),
)?;
```

`BlockBuilder::new` takes a `BlockName` (typically a `StaticBlockName` constant). The returned `BuiltBlock` contains the block ID and associated item ID.

## Textures

Textures are identified by name strings. Two types exist:

- **`StaticTextureName`** — a `Copy` newtype over `&'static str`. Use this for compile-time constant names (the common case).
- **`OwnedTextureName`** — a newtype over `String`. Use this when the name is computed at runtime (e.g., programmatically generated color names, or names built from a dynamic prefix).

Both implement the `TextureName` trait and can be passed anywhere a texture is expected.

### Declaring and registering a texture from a file

The typical pattern, taken from `basic_blocks.rs`:

```rust
// 1. Declare the name as a module-level constant:
pub const DIRT_TEXTURE: StaticTextureName = StaticTextureName("default:dirt");

// 2. Register the image data during initialization (usually in a setup fn):
include_texture_bytes!(game_builder, DIRT_TEXTURE, "textures/dirt.png")?;
```

`include_texture_bytes!` is a convenience macro that calls `game_builder.register_texture_bytes(tex_name, include_bytes!(file_name))`. The file path is resolved relative to the current source file (same semantics as `include_bytes!`).

### Using OwnedTextureName for dynamic names

When the texture name can't be a `'static` string — e.g., it's assembled at runtime from a variant value or a config string — use `OwnedTextureName`:

```rust
// Programmatically constructed name (must still be registered before use):
let tex = OwnedTextureName(format!("myplugin:lamp_{}", color));
game_builder.register_texture_bytes(&tex, &png_bytes)?;
```

`StaticTextureName` converts into `OwnedTextureName` via `From`, so you can `.into()` freely.

### CSS color placeholder

When no texture asset is available yet, generate a solid-color placeholder without any image file:

```rust
// No registration needed — CSS colors are generated on the fly.
OwnedTextureName::from_css_color("#ff00ff")   // bright magenta
OwnedTextureName::from_css_color("rgb(120 120 120)")
OwnedTextureName::from_css_color("orange")    // named CSS colors also work
```

## Appearance

Important policy: This project welcomes LLM-generated code, but does not permit ML-generated textures or media assets. If the necessary texture is unavailable, use either an existing texture,
a placeholder generated using a *command-line imagemagick call or similar*, or a CSS color using `OwnedTextureName::from_css_color("#ff00ff")`. Do not call a diffusion model or other AI image generator under any circumstances.

A common fallback pattern for placeholder textures is a bright magenta, or inverted version of an existing texture.

Pick exactly one appearance method:

### Cube (simple, same texture all faces)
```rust
.set_cube_single_texture(TEXTURE)
```

### Cube (per-face textures)
```rust
.set_cube_appearance(
    CubeAppearanceBuilder::new()
        .set_individual_textures(left, right, top, bottom, front, back)
        // optionally:
        .set_needs_transparency()   // for textures with alpha cutout
        .set_needs_translucency()   // for textures with partial transparency
        .set_rotate_laterally()     // rotates with player facing direction on place
)
```

### Plant-like (crossed-planes, like grass/flowers)
```rust
.set_plant_like_appearance(
    PlantLikeAppearanceBuilder::new()
        .set_texture(TEXTURE)
        .set_wave_effect_scale(0.5)  // 0.0 disables waving
        .set_is_solid(false)         // default false
)
```
Plant-like blocks typically also need `.set_allow_light_propagation(true)`.

### Axis-aligned boxes (custom geometry)
```rust
.set_axis_aligned_boxes_appearance(
    AxisAlignedBoxesAppearanceBuilder::new()
        .add_box(
            // if rotation_mode is RotateHorizontally, block will rotate based on player facing direction when placing.
            // This is typically what you want, except for blocks that have a global orientation
            // no matter which way a player is facing (e.g. on-the-ground compass block that must
            // indicate north in global coordinates)
            AaBoxProperties::new_single_tex(TEXTURE, TextureCropping::AutoCrop, RotationMode::RotateHorizontally),
            /* x= */ (-0.5, 0.5),
            /* y= */ (-0.5, 0.5),
            /* z= */ (-0.5, 0.5),
        )
        // Add more boxes as needed:
        .add_box_with_variant_mask(box_props, x, y, z, variant_mask)
        // variant_mask selects which variant values show this box (bitwise AND check)
)
```

`AaBoxProperties` constructors:
- `new_single_tex(texture, crop_mode, rotation_mode)` — same texture all faces
- `new(left, right, top, bottom, front, back, crop_mode, rotation_mode)` — per-face
- `new_custom_usage(...)` — control `is_visible`, `is_colliding`, `is_tool_hitbox` independently
- `new_plantlike(texture, rotation_mode)` — crossed-plane appearance within a box

As of 2026, due to LLM limitations in spatial reasoning, consider generating placeholder boxes and asking the user to refine them iteratively while testing in-game.

## Item and Display

```rust
.set_display_name("Human Readable Name")
.set_inventory_texture(TEXTURE)          // if different from block texture
.set_item_sort_key("namespace:category:name")
.add_item_group(some_item_group)
```

## Block Groups and Diggability

Block groups control tool effectiveness and other game logic:

```rust
.add_block_group(block_groups::STONE)    // affects dig speed with pickaxe
.add_block_group(block_groups::FIBROUS)  // affected by axe
.add_block_groups([GROUP_A, GROUP_B])    // add multiple at once
.set_not_diggable()
.set_wear_multiplier(2.0)               // tools wear faster on this block
```

`set_matter_type(MatterType::...)` also adds the corresponding block group automatically:
- `MatterType::Solid` (default)
- `MatterType::Liquid` — also call `.set_liquid_flow(Some(duration))` if you need automatic flow.
- `MatterType::Gas`

## Physics / Behavior

```rust
.set_allow_light_propagation(true)   // light passes through (needed for plant-like, leaves, etc.)
.set_light_emission(4)               // 0–15, causes glow; 15 = full brightness
.set_falls_down(true)                // falls like sand/gravel
.set_trivially_replaceable(true)     // air-like; placing other blocks replaces this
.set_liquid_flow(Some(Duration::from_secs(1)))
.set_footstep_sound(Some(sound_key))
```

## Dropped Items

BlockBuilder automatically defines an item corresponding to the block. Placing that item
will place the block at the given location.

By default the block drops that automatically-generated item. Override with:

```rust
.set_simple_dropped_item(ITEM_NAME.0, count)
.set_no_drops()
.set_dropped_item(DroppedItem::...)
.set_dropped_item_closure_extended(|param| (ITEM_NAME, count))  // randomized/variant-based drops
```

For example, dirt-with-grass has a corresponding dirt-with-grass item available to creative-mode players, but because digging it destroys the grass, it drops a normal dirt block.

## Interactions

```rust
.add_interact_key_menu_entry("internal_name", "Display Label")
// To handle the interaction, use add_modifier to set block.interact_key_handler
```

## Advanced

### LOD (Far Geometry) Color

This is an optional override. If unset, the block's texture will be automatically analyzed.

```rust
.override_lod_colors(0xffrrggbb_top, 0xffrrggbb_side, orientation_bias)
// orientation_bias > 0 biases toward top texture; < 0 biases toward side texture
.set_lod_orientation_bias(0.5)
```

### Extended Data and Variants

These require additional logic to produce gameplay effects.

```rust
.set_extended_data_initializer(Box::new(|ctx, coord, tool, | {
    // return Some(ExtendedData) or None
}))
.set_extra_variant_func(Box::new(|ctx, coord, tool, _| {
    // return variant u32 to modify the placed variant
}))
```

Consult the `advanced_block_features` skill document.

### add_modifier and add_item_modifier (Escape Hatches)

These are placeholders for functionality not yet exposed as dedicated builder methods. They run just before registration and provide direct access to the underlying structs.

```rust
.add_modifier(|block: &mut BlockType| {
    // Set handlers not yet exposed as builder methods:
    block.interact_key_handler = Some(Box::new(move |ctx, coord, menu_entry| { ... }));
    block.dig_handler_full = Some(Box::new(move |ctx, coord, tool| { ... }));
    block.tap_handler_full = Some(Box::new(move |ctx, coord, tool| { ... }));
    block.step_on_handler_full = Some(Box::new(move |coord, player| { ... }));
    // Override physics info:
    block.client_info.physics_info = Some(PhysicsInfo::Air(Empty {}));
})

.add_item_modifier(|item: &mut Item| {
    // Customize item properties not yet exposed:
    item.place_handler = Some(Box::new(move |ctx, coord, anchor, tool| { ... }));
})
```

Common fields set via `add_modifier`:
- `block.interact_key_handler` — called when player presses interact key
- `block.dig_handler_{full, inline}` — called when block is dug
- `block.tap_handler_{full, inline}` — called on a tap/hit
- `block.step_on_handler` — called when a player steps on the block
- `block.client_info.physics_info` — override physics (e.g., `PhysicsInfo::Air` for pass-through)

If writing any handlers, consult the `advanced_block_features` skill document first.

## Worked Examples

### Simple stone-like block
```rust
BlockBuilder::new(MY_BLOCK)
    .add_block_group(block_groups::STONE)
    .set_display_name("My Block")
    .set_cube_single_texture(MY_BLOCK_TEX)
```

### Leaves (transparent, light-propagating)
```rust
BlockBuilder::new(MY_LEAVES)
    .add_block_group(block_groups::FIBROUS)
    .add_block_group(block_groups::TREE_LEAVES)
    .set_cube_appearance(
        CubeAppearanceBuilder::new()
            .set_single_texture(MY_LEAVES_TEX)
            .set_needs_transparency(),
    )
    .set_allow_light_propagation(true)
```

### Flower (plant-like, passable)
```rust
BlockBuilder::new(MY_FLOWER)
    .set_plant_like_appearance(
        PlantLikeAppearanceBuilder::new().set_texture(MY_FLOWER_TEX),
    )
    .set_display_name("My Flower")
    .set_inventory_texture(MY_FLOWER_TEX)
    .set_allow_light_propagation(true)
    .add_modifier(|block: &mut BlockType| {
        block.client_info.physics_info = Some(PhysicsInfo::Air(Empty {}));
    })
```

### Glowing block
```rust
BlockBuilder::new(MY_LAMP)
    .set_cube_single_texture(MY_LAMP_TEX)
    .set_light_emission(15)
    .set_display_name("My Lamp")
```

### Interactive block with axis-aligned boxes
```rust
BlockBuilder::new(MY_DEVICE)
    .set_axis_aligned_boxes_appearance(
        AxisAlignedBoxesAppearanceBuilder::new()
            .add_box(
                AaBoxProperties::new_single_tex(MY_TEX, TextureCropping::NoCrop, RotationMode::RotateHorizontally),
                (-0.5, 0.5), (-0.5, 0.0), (-0.5, 0.5),
            ),
    )
    .set_allow_light_propagation(true)
    .set_display_name("My Device")
    .add_interact_key_menu_entry("", "Use")
    .add_modifier(|block: &mut BlockType| {
        block.interact_key_handler = Some(Box::new(move |ctx, coord, _| {
            // handle interaction
            Ok(None)
        }));
    })
```
