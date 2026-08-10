//! A small CLI for exercising the server's debug RPC interface
//! (`perovskite_core::protocol::debug`), intended to be friendly for both humans and
//! LLM/agent-driven usage.
//!
//! Start a server with the debug interface enabled, e.g.:
//!   cargo run -p perovskite_game_api --release --features=server,discord -- \
//!       --data-dir=/tmp/foo --enable_debug_interface --port=28273
//!
//! Then, from another terminal (defaults to grpc://localhost:28273, i.e. a plaintext connection
//! to a server started as above; pass e.g. --endpoint=grpcs://example.com:28273 for TLS):
//!   cargo run -p perovskite_server --bin debug_client -- find-block-defs '^default:dirt'
//!   cargo run -p perovskite_server --bin debug_client -- find-item-defs '.*'
//!   cargo run -p perovskite_server --bin debug_client -- last-events
//!   cargo run -p perovskite_server --bin debug_client -- set-working-coord 0 0 0
//!   cargo run -p perovskite_server --bin debug_client -- get-block 0 0 0
//!   cargo run -p perovskite_server --bin debug_client -- set-block 0 0 0 default:dirt
//!   cargo run -p perovskite_server --bin debug_client -- get-block-by-id 4096
//!   cargo run -p perovskite_server --bin debug_client -- dig-block 0 0 0
//!
//! Adding a new debug RPC? Add a variant to `Command` below, a matching arm in `main`, and a
//! `run_*` function alongside `run_find_block_defs`/`run_find_item_defs`. Output is always
//! printed as `{:#?}` (Rust's pretty-printed Debug format) rather than a custom format, so new
//! RPCs don't need any additional formatting code.
//!
//! Not yet implemented (can be added later): proto reflection (to discover RPCs/fields without
//! recompiling this client) and field masks (to request a subset of fields in the response).

use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use perovskite_core::protocol::debug::{
    perovskite_debug_client::PerovskiteDebugClient, DigBlockReq, FindBlockDefsReq, FindItemDefsReq,
    GetBlockByIdReq, GetBlockReq, LastEventsReq, SetBlockReq, SetWorkingCoordReq,
};
use tonic::transport::{Channel, ClientTlsConfig};
use tonic::Status;

#[derive(Parser)]
#[command(
    name = "debug_client",
    about = "CLI for Perovskite's debug RPC interface"
)]
struct Cli {
    /// Address of the server's debug interface. Use "grpc://" for a plaintext connection
    /// (matches a server run without a tls.ron in its data dir) or "grpcs://" for TLS (native +
    /// webpki root certificates are trusted automatically); "http://"/"https://" also work.
    #[arg(long, default_value = "grpc://localhost:28273")]
    endpoint: String,

    #[command(subcommand)]
    command: Command,

    #[arg(long, default_value = "false")]
    verbose: bool,
}

/// Rewrites a `grpc://`/`grpcs://` endpoint into the `http://`/`https://` URI tonic expects,
/// returning whether TLS should be used.
fn resolve_endpoint(endpoint: &str) -> Result<(String, bool)> {
    if let Some(rest) = endpoint.strip_prefix("grpcs://") {
        Ok((format!("https://{rest}"), true))
    } else if let Some(rest) = endpoint.strip_prefix("grpc://") {
        Ok((format!("http://{rest}"), false))
    } else if let Some(rest) = endpoint.strip_prefix("https://") {
        Ok((format!("https://{rest}"), true))
    } else if let Some(rest) = endpoint.strip_prefix("http://") {
        Ok((format!("http://{rest}"), false))
    } else {
        bail!("endpoint {endpoint:?} must start with grpc://, grpcs://, http://, or https://")
    }
}

#[derive(Subcommand)]
enum Command {
    /// Find block definitions whose short_name matches a regex, e.g. "^default:" or ".*".
    FindBlockDefs {
        /// Regular expression (Rust `regex` crate syntax) matched against each block's
        /// short_name.
        regex: String,
    },
    /// Find item definitions whose short_name matches a regex, e.g. "^default:" or ".*".
    FindItemDefs {
        /// Regular expression (Rust `regex` crate syntax) matched against each item's
        /// short_name.
        regex: String,
    },
    /// Fetch the last N events observed by the server's debug fake player (defaults to 64 if
    /// omitted).
    LastEvents {
        /// Maximum number of most-recent events to return.
        n: Option<i32>,
    },
    /// Move the debug fake player to the given block coordinate, which also becomes the center
    /// of the chunk-loading area used by other debug RPCs.
    SetWorkingCoord { x: i32, y: i32, z: i32 },
    /// Look up a block type by its raw numeric block ID (type + variant bits).
    GetBlockById {
        /// Raw block ID, as reported by e.g. `get-block`.
        block_id: u32,
    },
    /// Get the block at a coordinate.
    GetBlock {
        x: i32,
        y: i32,
        z: i32,
        /// If true, also return the extended data for this block, if any.
        #[arg(long)]
        extended_data: bool,
    },
    /// Set the block at a coordinate, by name and variant. Runs no dig/place handlers and does
    /// not check placement rules; this is a low-level overwrite intended for local
    /// development/iteration.
    SetBlock {
        x: i32,
        y: i32,
        z: i32,
        /// Block short name, e.g. "default:dirt".
        block_name: String,
        /// Variant to set (defaults to 0 if omitted).
        #[arg(default_value_t = 0)]
        variant: u32,
    },
    /// Dig the block at a coordinate, running its dig handler as if a player dug it.
    DigBlock {
        x: i32,
        y: i32,
        z: i32,
        /// Item short name of the tool to dig with. If omitted, the server digs with a
        /// built-in tool that instantly digs any solid block without wear.
        #[arg(long)]
        item: Option<String>,
        /// Quantity of the tool item stack to make before digging.
        #[arg(long, default_value_t = 1)]
        quantity: u32,
        /// If true, also return the extended data for the block before/after digging, if any.
        #[arg(long)]
        extended_data: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let (uri, use_tls) = resolve_endpoint(&cli.endpoint)?;
    let mut builder = Channel::from_shared(uri.clone())
        .with_context(|| format!("{uri:?} is not a valid server address"))?
        .connect_timeout(Duration::from_secs(10));
    if use_tls {
        let tls = ClientTlsConfig::new()
            .with_native_roots()
            .with_webpki_roots();
        builder = builder.tls_config(tls)?;
    }
    let channel = builder
        .connect()
        .await
        .with_context(|| format!("Failed to connect to debug server at {uri}"))?;
    let mut client = PerovskiteDebugClient::new(channel);

    match cli.command {
        Command::FindBlockDefs { regex } => {
            run_find_block_defs(&mut client, regex, cli.verbose).await
        }
        Command::FindItemDefs { regex } => {
            run_find_item_defs(&mut client, regex, cli.verbose).await
        }
        Command::LastEvents { n } => run_last_events(&mut client, n).await,
        Command::SetWorkingCoord { x, y, z } => run_set_working_coord(&mut client, x, y, z).await,
        Command::GetBlockById { block_id } => run_get_block_by_id(&mut client, block_id).await,
        Command::GetBlock {
            x,
            y,
            z,
            extended_data,
        } => run_get_block(&mut client, x, y, z, cli.verbose, extended_data).await,
        Command::SetBlock {
            x,
            y,
            z,
            block_name,
            variant,
        } => run_set_block(&mut client, x, y, z, block_name, variant).await,
        Command::DigBlock {
            x,
            y,
            z,
            item,
            quantity,
            extended_data,
        } => {
            run_dig_block(
                &mut client,
                x,
                y,
                z,
                item,
                quantity,
                cli.verbose,
                extended_data,
            )
            .await
        }
    }
}

async fn run_find_block_defs(
    client: &mut PerovskiteDebugClient<Channel>,
    regex: String,
    verbose: bool,
) -> Result<()> {
    println!("FindBlockDefs(name_regex = {regex:?})");
    match client
        .find_block_defs(FindBlockDefsReq { name_regex: regex })
        .await
    {
        Ok(resp) => {
            let entries = resp.into_inner().entries;
            println!("OK: {} block definition(s) matched", entries.len());
            for entry in entries {
                if verbose {
                    println!("---\n{entry:#?}");
                } else {
                    println!("{}", entry.name());
                }
            }
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

async fn run_find_item_defs(
    client: &mut PerovskiteDebugClient<Channel>,
    regex: String,
    verbose: bool,
) -> Result<()> {
    println!("FindItemDefs(name_regex = {regex:?})");
    match client
        .find_item_defs(FindItemDefsReq { name_regex: regex })
        .await
    {
        Ok(resp) => {
            let entries = resp.into_inner().entries;
            println!("OK: {} item definition(s) matched", entries.len());
            for entry in entries {
                if verbose {
                    println!("---\n{entry:#?}");
                } else {
                    println!("{}", entry.name());
                }
            }
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

async fn run_last_events(
    client: &mut PerovskiteDebugClient<Channel>,
    n: Option<i32>,
) -> Result<()> {
    println!("LastEvents(n = {n:?})");
    match client.last_events(LastEventsReq { n }).await {
        Ok(resp) => {
            let events = resp.into_inner().events;
            println!("OK: {} event(s)", events.len());
            for event in events {
                println!("---\n{event}");
            }
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

async fn run_set_working_coord(
    client: &mut PerovskiteDebugClient<Channel>,
    x: i32,
    y: i32,
    z: i32,
) -> Result<()> {
    println!("SetWorkingCoord(x = {x}, y = {y}, z = {z})");
    match client
        .set_working_coord(SetWorkingCoordReq { x, y, z })
        .await
    {
        Ok(_) => {
            println!("OK");
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

async fn run_get_block_by_id(
    client: &mut PerovskiteDebugClient<Channel>,
    block_id: u32,
) -> Result<()> {
    println!("GetBlockById(block_id = 0x{block_id:x})");
    match client.get_block_by_id(GetBlockByIdReq { block_id }).await {
        Ok(resp) => {
            println!("OK: {:#?}", resp.into_inner());
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

/// Prints a `GetBlockResp` in either verbose (full proto) or terse (one-line description, plus
/// extended data if requested) form. Shared between `get-block` and `dig-block` (which reports
/// the block before and after digging using the same response type).
fn print_get_block_resp(
    resp: &perovskite_core::protocol::debug::GetBlockResp,
    verbose: bool,
    extended_data: bool,
) {
    if verbose {
        println!("{:#?}", resp);
    } else if extended_data {
        println!(
            "{} \t{}",
            resp.description,
            resp.extended_data
                .as_deref()
                .unwrap_or("<no extended data>")
        );
    } else {
        println!("{}", &resp.description);
    }
}

async fn run_get_block(
    client: &mut PerovskiteDebugClient<Channel>,
    x: i32,
    y: i32,
    z: i32,
    verbose: bool,
    extended_data: bool,
) -> Result<()> {
    println!("GetBlock(x = {x}, y = {y}, z = {z})");
    match client
        .get_block(GetBlockReq {
            x,
            y,
            z,
            extended_data,
        })
        .await
    {
        Ok(resp) => {
            print_get_block_resp(&resp.into_inner(), verbose, extended_data);
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

async fn run_set_block(
    client: &mut PerovskiteDebugClient<Channel>,
    x: i32,
    y: i32,
    z: i32,
    block_name: String,
    variant: u32,
) -> Result<()> {
    println!(
        "SetBlock(x = {x}, y = {y}, z = {z}, block_name = {block_name:?}, variant = 0x{variant:x})"
    );
    match client
        .set_block(SetBlockReq {
            x,
            y,
            z,
            block_name,
            variant,
        })
        .await
    {
        Ok(resp) => {
            println!("OK: {:#?}", resp.into_inner());
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

async fn run_dig_block(
    client: &mut PerovskiteDebugClient<Channel>,
    x: i32,
    y: i32,
    z: i32,
    item: Option<String>,
    quantity: u32,
    verbose: bool,
    extended_data: bool,
) -> Result<()> {
    println!("DigBlock(x = {x}, y = {y}, z = {z}, item = {item:?}, quantity = {quantity})");
    match client
        .dig_block(DigBlockReq {
            x,
            y,
            z,
            item_name: item,
            quantity: Some(quantity),
            extended_data,
        })
        .await
    {
        Ok(resp) => {
            let resp = resp.into_inner();
            if let Some(before) = &resp.before {
                print!("Before: ");
                print_get_block_resp(before, verbose, extended_data);
            }
            if let Some(after) = &resp.after {
                print!("After:  ");
                print_get_block_resp(after, verbose, extended_data);
            }
            if verbose {
                println!("Item stacks: {:#?}", resp.item_stacks);
            } else {
                println!(
                    "Item stacks: {}",
                    resp.item_stacks
                        .iter()
                        .map(|s| format!("{}x{}", s.quantity, s.item_name))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if let Some(tool_wear) = resp.tool_wear {
                println!("Tool wear: {tool_wear}");
            }
            Ok(())
        }
        Err(status) => {
            print_rpc_error(&status);
            Ok(())
        }
    }
}

/// Prints an RPC error in a format that's easy for both humans and LLMs/agents to parse.
/// Covers both ordinary returned errors (e.g. an invalid regex) and errors caused by a
/// server-side panic, which tonic surfaces the same way.
fn print_rpc_error(status: &Status) {
    println!("ERROR: RPC failed");
    println!("  code: {:?}", status.code());
    println!("  message: {}", status.message());
}
