---
name: game_input_demo
description: Ask the user to demonstrate in-game actions via the dev server when you need sample gameplay input, in-game builds, or other game input you cannot determine from code or prose text alone.
---

Use this skill when you are stuck because you need information that can only come from actually running the game — for example: a set of blocks/actions the user describes, the item name in a particular hotbar slot, the interaction target of a block they want to configure, or a sequence of actions to understand expected behavior.

## When to invoke

- "Which block are you referring to?" → you need XYZ coordinates
- "What item do you want the machine to accept?" → you need an item name string
- "Show me how you interact with it" → you need action type + target
- Any time you need ground-truth game data that isn't derivable from source code

## Step 1: Decide what initial state the player needs

Before running the server, determine:
- What items should the player have in their hotbar?
- Does the world need specific blocks pre-placed?
- Should spawn be adjusted to a particular location?

Write this down — you'll use it to fill in the SCENE SETUP section, if you have any *a priori* setup the user should see.

## Step 2: Modify the SCENE SETUP section in dev_server.rs

`perovskite_game_api/src/bin/dev_server.rs` contains a clearly marked `SCENE SETUP` block in `main()`, between `game.set_flatland_mapgen(dirt)` and `game.add_command("stop", ...)`. Put all per-session customizations there.

### Override spawn location

```rust
use cgmath::vec3; // add at top of file

// inside SCENE SETUP:
game.server_builder_mut().game_behaviors_mut().spawn_location =
    Box::new(|_player_name| vec3(0.0, 5.0, 0.0));
```

### Give items to a player on first connect

Add these imports at the top of the file:

```rust
use perovskite_server::game_state::{
    game_behaviors::GenericAsyncHandler,
    // ... existing imports ...
};
use perovskite_core::protocol::items::item_stack::QuantityType;
use perovskite_server::game_state::items::ItemStack;
```

Define a handler struct (outside `main`):

```rust
struct GiveItemsOnJoin {
    items: Vec<(String, u32)>, // (item_name, quantity)
}

#[async_trait]
impl GenericAsyncHandler<Player, ()> for GiveItemsOnJoin {
    async fn handle(&self, player: &Player, context: HandlerContext<'_>) -> Result<()> {
        let inv_key = player.main_inventory();
        for (item_name, qty) in &self.items {
            context
                .inventory_manager()
                .mutate_inventory_atomically(&inv_key, |inv| {
                    let stack = ItemStack {
                        proto: perovskite_core::protocol::items::ItemStack {
                            item_name: item_name.clone(),
                            quantity: *qty,
                            current_wear: 0,
                            quantity_type: Some(QuantityType::Stack(256)),
                        },
                    };
                    inv.try_insert(stack);
                    Ok(())
                })?;
        }
        Ok(())
    }
}
```

Register it inside SCENE SETUP:

```rust
game.server_builder_mut().game_behaviors_mut().on_player_join =
    Box::new(GiveItemsOnJoin {
        items: vec![
            ("default_game:tools/steel_pick".to_string(), 1),
            ("default_game:basic_blocks/dirt".to_string(), 64),
        ],
    });
```

Item name strings follow the pattern `{crate_name}:{category}/{item_name}`. Use the item names logged by the dev server itself, or search with `grep -r 'item_name:' perovskite_game_api/src/`.

## Step 3: Run the dev server

From the workspace root (`C:\cuberef`):

```
cargo run --bin dev_server --features="default_game,server,test-support" -- --output dev_server_recording.txt
```

This command **blocks** until the server shuts down. The server listens on port **28275**.

The recording file will be created at `C:\cuberef\dev_server_recording.txt`. The absolute path is also printed to stdout at startup.

## Step 4: Give the user these instructions

Provide a copy-pasteable message. Be specific about what actions to perform.

Example:

> Please connect to the dev server at **localhost:28275** (register with any username and password — the account is temporary). Once you're in, **[describe exactly what to do]**. When you're done, type `/stop` in the chat to shut down the server.

Tips:
- Tell the user which items they'll have (if you set up scene state in step 2)
- Be explicit: "dig the block you want to use as the input", "interact (right-click) with the machine", etc.
- Remind them to type `/stop` when finished — this is the only way to cleanly exit

## Step 5: Read and parse the recording

After the server exits, read `C:\cuberef\dev_server_recording.txt`.

**File structure:**
```
SESSION_START timestamp=2025-05-14T10:30:00
ts=10:30:05 action=dig player=alice item_slot=0 item=default_game:tools/steel_pick:1 target=block(3,-1,7) player_pos=(3.25,-0.00,7.00) az=180 el=-15
ts=10:30:08 action=interact player=alice item_slot=2 item=empty target=block(0,0,0) player_pos=(1.00,1.00,1.00) az=0 el=0
SESSION_END timestamp=2025-05-14T10:30:12
```

**Field reference:**
| Field | Values |
|-------|--------|
| `action` | `dig`, `tap`, `place`, `interact` |
| `item_slot` | 0-based hotbar index |
| `item` | `{item_name}:{quantity_or_wear}`, `empty`, or `no_inventory` |
| `target` | `block(x,y,z)`, `entity(id)`, or `none` |
| `player_pos` | `(x,y,z) az={deg} el={deg}` — position rounded to 0.25 blocks, angles to 15° |

For block coordinate lookups: the `target=block(x,y,z)` of a **dig** or **interact** action is the block the player acted on. For a **place** action, use `anchor=block(x,y,z)` (the adjacent face) to infer where the new block was placed.

## Cleanup

The SCENE SETUP block should be reverted to its default (empty/commented-out) state after the demo session, unless the setup is reusable for the feature being built.
