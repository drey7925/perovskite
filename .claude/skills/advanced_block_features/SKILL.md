---
name: advanced_block_features
description: Advanced block functionality: event handlers (dig, tap, step-on, interact), extended data (simple_data, inventories, custom protobuf data), client extended data, and block mutation APIs. Use alongside define_new_block. Does not cover timers or circuits.
---

This skill covers the advanced/stateful side of blocks. For basic BlockBuilder usage (appearance,
drops, block groups, physics), see `define_new_block`. The raw types live in
`perovskite_server/src/game_state/blocks.rs`; the builder wrappers are in
`perovskite_game_api/src/blocks.rs`.

Real-world examples used as references:
- `perovskite_game_api/src/default_game/chest.rs` — interact handler, extended data initializer, inventory popup
- `perovskite_game_api/src/default_game/signs.rs` — client extended data, text rendering, interact popup

---

## Handler Overview: Full vs Inline

Several handler slots exist in both a `_full` and `_inline` variant. Both are set via `add_modifier`.

| Variant | Signature base | Atomicity | Map access |
|---------|---------------|-----------|------------|
| `_inline` | `InlineContext, &mut BlockId, &mut ExtendedDataHolder, Option<&ItemStack>` | Atomic on the target block | Block types + item manager only |
| `_full` | `&HandlerContext, BlockCoordinate, Option<&ItemStack>` | Racy — block may have changed | Full game state (map, inventories, popups, etc.) |

**Both handlers run if both are set**; their `BlockInteractionResult` item stacks are merged.

Use inline when you need to atomically read/write the block or its extended data.
Use full when you need to access the broader game map or create a popup.

Full handlers are racy - the block may have changed between the time the event was queued and the time the handler runs. However, if the handler
then calls map functions, it still enjoys all of the atomicity guarantees of the chosen function.

### Return type

Most interaction handlers return `Result<BlockInteractionResult>`:

```rust
pub struct BlockInteractionResult {
    pub item_stacks: Vec<ItemStack>,  // items given to the player
    pub tool_wear: u32,               // added to tool wear
}
// Default::default() = no drops, no wear — the common return for side-effect-only handlers
```

---

## Handler Slots

When using the perovskite_game_api BlockBuilder, set all of these via `add_modifier(|block: &mut BlockType| { ... })`.

If making a block directly using the low-level server API, assign directly to the fields on `BlockType`.

### Dig handlers — called when the block is fully dug

The default dig handler (built from `DroppedItem`) removes the block and returns drops.
**If you override `dig_handler_inline`, you are responsible for removing the block** (set `*block_id = AIR_ID`).

```rust
// Inline — atomic, good for state machines
block.dig_handler_inline = Some(Box::new(|ctx, block_id, ext, tool| {
    *block_id = AIR_ID;
    ext.clear(); // drop extended data
    Ok(BlockInteractionResult {
        item_stacks: vec![/* ... */],
        tool_wear: 0,
    })
}));

// Full — racy, good for side effects that need map access
block.dig_handler_full = Some(Box::new(|ctx, coord, tool| {
    // block may already be gone; coord tells you where it was
    Ok(Default::default())
}));
```

When overriding these handlers, you are responsible for computing the tool wear.
The easiest way to do this is to use the `dig_time_and_wear` function of the item manager,
typically accessed via `ctx.items().dig_time_and_wear(stack, block_type)`.

`dig_time_and_wear` correctly handles when the item stack is `None` (i.e. the player is digging with their hand).

This function returns `Some((duration, wear))` if the item can dig the block, and `None` otherwise.
The duration is probably unimportant for most handlers, since durations are handled at the client side.

### Tap handlers — called when the block is hit but not dug

Same signature as dig handlers. Both run even if the block is not removed.
`tool_wear` in the return value is applied for taps too.

```rust
block.tap_handler_inline = Some(Box::new(|ctx, block_id, ext, tool| {
    // e.g. play a sound, update state
    Ok(Default::default())
}));
block.tap_handler_full = Some(Box::new(|ctx, coord, tool| {
    Ok(Default::default())
}));
```

### Step-on handlers — called when a player steps on the block

Best-effort: **some events will be lost**. Player position is potentially stale.
`tool_wear` in the return value is ignored. The `ItemStack` argument is always `None`.

```rust
block.step_on_handler_inline = Some(Box::new(|ctx, block_id, ext, _tool| {
    Ok(Default::default())
}));
block.step_on_handler_full = Some(Box::new(|ctx, coord, _tool| {
    Ok(Default::default())
}));
```

If a precise, synchronous step-on handler is needed, this should be discussed via Discord/GitHub.
This functionality is under consideration, but so far does not have a concrete use-case.

### Interact key handler — called when the player presses the interact key

Returns `Option<Popup>`. Use with `add_interact_key_menu_entry` to show a label in the
interact menu. If multiple named entries exist, `menu_entry` tells you which was chosen.

```rust
.add_interact_key_menu_entry("", "Open")  // "" = unnamed default
.add_modifier(|block| {
    block.interact_key_handler = Some(Box::new(|ctx, coord, menu_entry| {
        match ctx.initiator() {
            EventInitiator::Player(p) => {
                // build and return a popup, or Ok(None) to do nothing visible
                Ok(Some(ctx.new_popup().title("My Block") /* ... */ .build()))
            }
            _ => Ok(None),
        }
    }));
})
```

Client selections

### place_upon_handler — NOT YET IMPLEMENTED

`BlockType::place_upon_handler` exists in the struct but is marked `NOT YET IMPLEMENTED`.
Do not use it. If it is needed, please request it via Discord/GitHub.

The plan is to use this to allow blocks to accept items placed upon them, but there are open
questions regarding whether item handlers or this block handler take precedence.

---

## Extended Data

`ExtendedData` is per-block server-side state. It is never sent to the client directly
(see Client Extended Data below for that path).

```rust
pub struct ExtendedData {
    pub custom_data: Option<CustomData>,             // Box<dyn Any + Send + Sync>
    pub simple_data: HashMap<String, String>,        // lightweight key-value
    pub inventories: hashbrown::HashMap<String, Inventory>,
}
```

### simple_data — key-value string pairs

Best for small amounts of text: owner names, flags, short config values.

```rust
// Reading (in a full handler, via game_map):
let (_, value) = ctx.game_map().get_block_with_extended_data(coord, |data| {
    Ok(data.simple_data.get("my_key").cloned())
})?;

// Writing (in an inline handler or mutate_block_atomically):
ctx.game_map().mutate_block_atomically(coord, |_block_id, ext| {
    let data = ext.get_or_insert_with(Default::default);
    data.simple_data.insert("my_key".to_string(), "my_value".to_string());
    Ok(())
})?;
```

### inventories — block inventories

The engine automatically persists and loads inventories; you only need to declare the
inventory's name/size. Create them in the extended data initializer or on first access.

```rust
// In initializer:
let mut data = ExtendedData::default();
data.inventories.insert("storage".to_string(), Inventory::new(4, 8)); // rows × cols
Ok(Some(data))

// In a popup:
ctx.new_popup()
    .title("Storage")
    .inventory_view_block(
        "view_id",
        "Contents:",
        (4, 8),      // rows × cols — must match inventory size
        coord,
        "storage".to_string(),  // inventory name in ExtendedData
        true,        // can player place items?
        true,        // can player take items?
        false,       // scrollable?
    )?
    // ... add player inventory view, build, etc.
```

### custom_data — arbitrary typed data

Use when `simple_data` is insufficient (e.g., complex state, numeric fields, nested structures).
Typically backed by a Protobuf message. Must be paired with serialize/deserialize handlers.

```rust
use prost::Message;

#[derive(Clone, prost::Message)]
pub struct MyBlockState {
    #[prost(uint32, tag = "1")]
    pub counter: u32,
    #[prost(string, tag = "2")]
    pub label: String,
}
```

Register the serialization pair via `add_modifier` and the protobuf-specific helper:

```rust
.add_modifier(|block| {
    block.register_proto_serialization_handlers::<MyBlockState>();
})
```

Accessing `custom_data` in an inline handler:

```rust
block.dig_handler_inline = Some(Box::new(|ctx, block_id, ext, tool| {
    if let Some(data) = ext.as_ref().and_then(|e| e.custom_data.as_ref()) {
        if let Some(state) = data.downcast_ref::<MyBlockState>() {
            // read state
        }
    }
    // To write — use DerefMut (sets dirty bit automatically):
    let data = ext.get_or_insert_with(Default::default);
    if let Some(s) = data.custom_data.as_mut().and_then(|d| d.downcast_mut::<MyBlockState>()) {
        s.counter += 1;
    }
    Ok(Default::default())
}));
```

> **[DATA LOSS RISK]** If `custom_data` uses interior mutability (Mutex, atomics,
> etc.) and you only access it via an *immutable* deref (e.g., `ext.deref()`, `&*ext`,
> `ext.as_ref()`), the dirty bit will NOT be set and changes will be silently lost when the
> chunk is unloaded. Call `ext.set_dirty()` explicitly in that case, or restructure to use
> `DerefMut` (i.e., `&mut *ext`) which always sets dirty.

---

## Extended Data Initializer

Called once when a player places the block. Use it to set up initial `simple_data`,
`inventories`, or `custom_data`.

```rust
.set_extended_data_initializer(Box::new(|ctx, pointee, stack| {
    //   ctx     : HandlerContext
    //   pointee : PointeeBlockCoords { selected: BlockCoordinate, preceding: Option<BlockCoordinate> }
    //   stack   : &ItemStack (the item being placed)
    //
    // Return Ok(None) to place with no extended data.
    // Return Ok(Some(ExtendedData { ... })) to attach data.
    Ok(ctx.initiator().player_name().map(|name| {
        let mut data = ExtendedData::default();
        data.simple_data.insert("owner".to_string(), name.to_string());
        data
    }))
}))
```

`ctx.initiator().player_name()` returns `Option<&str>` — None if placed by non-player.

Note that the extended data initializer is implemented via the default place handler. If you override the
place handler, you will need to initialize the extended data yourself when calling set_block or similar to actually place the block.

---

## Mutating Blocks from Full Handlers

From a full handler (which has `HandlerContext`), use `ctx.game_map()` for block operations.
`HandlerContext` derefs to `GameState`, so `ctx.game_map()` works directly.

```rust
// Read block + extended data atomically:
let (block_id, my_value) = ctx.game_map().get_block_with_extended_data(coord, |data| {
    Ok(data.simple_data.get("key").cloned())
})?;

// Atomically mutate block + extended data (takes a write lock on the chunk):
ctx.game_map().mutate_block_atomically(coord, |block_id, ext| {
    let data = ext.get_or_insert_with(Default::default);
    data.simple_data.insert("key".to_string(), "val".to_string());
    // Optionally change the block type:
    // *block_id = some_other_block_id;
    Ok(())
})?;

// Non-blocking variant — returns Ok(None) if chunk is not immediately available:
ctx.game_map().try_mutate_block_atomically(coord, |block_id, ext| {
    Ok(())
}, /*wait_for_inner=*/false)?;

// Just read the block ID (no extended data):
let id: BlockId = ctx.game_map().get_block(coord)?;
```

> **IMPORTANT:** `mutate_block_atomically` holds a **write lock on the chunk** for its
> duration. Keep the closure fast. Do not call `game_map` methods that could try to acquire a
> write lock on the same chunk from within the closure — that will deadlock.

### Deferred execution

To do work after the lock is released, or to avoid blocking in a timer:

```rust
ctx.run_deferred(|ctx| {
    ctx.game_map().mutate_block_atomically(coord, |_, ext| { Ok(()) })
});
// run_deferred_delayed is available for time-delayed side effects (avoid for most block logic)
```

---

## Client Extended Data

Lets the server send per-block data to the client for rendering (e.g., sign text).
Must opt in via `block.client_info.has_client_extended_data = true` and implement
`make_client_extended_data`.

```rust
.add_modifier(|block| {
    block.client_info.has_client_extended_data = true;
    block.make_client_extended_data = Some(Box::new(|ctx, ext_data| {
        // ctx: InlineContext (restricted — block types and items only, no game_map)
        // ext_data: &ExtendedData
        // Returns Ok(None) to send no client data; Ok(Some(...)) to send data.
        Ok(Some(ClientExtendedData {
            offset_in_chunk: 0,  // engine overwrites this; set to 0
            block_text: ext_data
                .simple_data
                .get("sign_text")
                .map(|t| BlockText { text: t.clone() })
                .into_iter()
                .collect(),
        }))
    }));
})
```

Imports needed:
```rust
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::protocol::render::BlockText;
```

> **Important:** The `make_client_extended_data` handler takes `InlineContext`, not
> `HandlerContext`. It cannot access the game map. It is called on a hot path (whenever the
> client needs the block data), so keep it fast.

---

## Variant-Based State (Lightweight Alternative to Extended Data)

Block variants are a 12-bit value stored inline with the block ID — zero allocation, zero
serialization. Useful for small counters (up to 4096 states), rotation, growth stages.

```rust
// In an inline handler:
let current_variant = block_id.variant();
let new_variant = (current_variant + 1) & 0xFFF;
*block_id = block_id.with_variant_unchecked(new_variant);  // no extended data write needed

// On placement, use set_extra_variant_func to pick an initial variant:
.set_extra_variant_func(Box::new(|ctx, pointee, stack, proposed_variant| {
    // Return the variant to use. proposed_variant is what the engine would use by default.
    Ok(proposed_variant)
}))
```

`add_box_with_variant_mask` on `AxisAlignedBoxesAppearanceBuilder` lets different variant
values show/hide geometry boxes without any extended data.

---

## Worked Examples

### Locked chest (extended data + ownership check)

```rust
const OWNER_KEY: &str = "myplugin:my_block:owner";

BlockBuilder::new(MY_LOCKED_BLOCK)
    .set_cube_single_texture(MY_TEX)
    .set_display_name("My Locked Block")
    .add_interact_key_menu_entry("", "Open")
    .set_extended_data_initializer(Box::new(|ctx, _pointee, _stack| {
        Ok(ctx.initiator().player_name().map(|name| {
            let mut data = ExtendedData::default();
            data.simple_data.insert(OWNER_KEY.to_string(), name.to_string());
            data
        }))
    }))
    .add_modifier(|block| {
        block.interact_key_handler = Some(Box::new(|ctx, coord, _menu_entry| {
            match ctx.initiator() {
                EventInitiator::Player(p) => {
                    let (_, owner) = ctx.game_map().get_block_with_extended_data(coord, |data| {
                        Ok(data.simple_data.get(OWNER_KEY).cloned())
                    })?;
                    let is_owner = owner.as_deref() == Some(p.player.name());
                    if is_owner {
                        Ok(Some(ctx.new_popup().title("Opened!").build()))
                    } else {
                        p.player.send_chat_message(
                            ChatMessage::new_server_message("You don't own this.")
                        )?;
                        Ok(None)
                    }
                }
                _ => Ok(None),
            }
        }));
    })
```

### Sign with rendered text (client extended data)

```rust
const TEXT_KEY: &str = "myplugin:sign:text";

BlockBuilder::new(MY_SIGN)
    // ... appearance ...
    .set_allow_light_propagation(true)
    .add_interact_key_menu_entry("", "Set text...")
    .add_modifier(|block| {
        block.client_info.has_client_extended_data = true;
        block.make_client_extended_data = Some(Box::new(|_ctx, ext| {
            Ok(Some(ClientExtendedData {
                offset_in_chunk: 0,
                block_text: ext.simple_data.get(TEXT_KEY)
                    .map(|t| BlockText { text: t.clone() })
                    .into_iter()
                    .collect(),
            }))
        }));
        block.interact_key_handler = Some(Box::new(|ctx, coord, _| {
            match ctx.initiator() {
                EventInitiator::Player(_) => {
                    // Build a popup with a text field and a save button.
                    // In the button callback, call mutate_block_atomically to update simple_data.
                    Ok(Some(build_sign_edit_popup(&ctx, coord)?))
                }
                _ => Ok(None),
            }
        }));
    })
```

### Counter block (variant-based, no extended data)

```rust
BlockBuilder::new(MY_COUNTER)
    .set_cube_single_texture(MY_TEX)
    .set_display_name("Counter")
    .add_modifier(|block| {
        block.tap_handler_inline = Some(Box::new(|_ctx, block_id, _ext, _tool| {
            let v = block_id.variant();
            *block_id = block_id.with_variant_unchecked((v + 1) & 0xF); // 0–15
            Ok(Default::default())
        }));
    })
```

---

## EventInitiator

Handlers receive context about who triggered the event:

```rust
match ctx.initiator() {
    EventInitiator::Player(p) => {
        let name: &str = p.player.name();
        p.player.send_chat_message(ChatMessage::new_server_message("Hello!"))?;
        p.player.has_permission(permissions::BYPASS_INVENTORY_CHECKS);
    }
    EventInitiator::Engine => { /* triggered by the engine itself */ }
    EventInitiator::Plugin(name) => { /* other non-player server-side action */ }
}
// Convenience shorthand:
ctx.initiator().player_name() // Option<&str>
```

`InlineContext` also has `initiator()` with the same shape.

---

## Key Imports

```rust
use perovskite_server::game_state::blocks::{
    BlockType, ExtendedData, ExtendedDataHolder, InlineContext, BlockInteractionResult,
    FastBlockName,
};
use perovskite_server::game_state::event::{EventInitiator, HandlerContext};
use perovskite_core::block_id::special_block_defs::AIR_ID;
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::protocol::render::BlockText;
use perovskite_server::game_state::client_ui::{Popup, UiElementContainer};
use perovskite_server::game_state::inventory::Inventory;
use perovskite_core::chat::ChatMessage;
use anyhow::{Context, Result};
use prost::Message; // for protobuf custom_data
```