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
    perovskite_debug_client::PerovskiteDebugClient, FindBlockDefsReq, FindItemDefsReq,
};
use tonic::transport::{Channel, ClientTlsConfig};
use tonic::Status;

#[derive(Parser)]
#[command(name = "debug_client", about = "CLI for Perovskite's debug RPC interface")]
struct Cli {
    /// Address of the server's debug interface. Use "grpc://" for a plaintext connection
    /// (matches a server run without a tls.ron in its data dir) or "grpcs://" for TLS (native +
    /// webpki root certificates are trusted automatically); "http://"/"https://" also work.
    #[arg(long, default_value = "grpc://localhost:28273")]
    endpoint: String,

    #[command(subcommand)]
    command: Command,
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
        bail!(
            "endpoint {endpoint:?} must start with grpc://, grpcs://, http://, or https://"
        )
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
    ///
    /// As of this writing, the server's FindItemDefs handler is a `todo!()` stub, so this
    /// command doubles as a way to validate that this client handles a server-side panic (which
    /// tonic surfaces to the client as a plain RPC error, not a crash) as gracefully as an
    /// ordinary returned error.
    FindItemDefs {
        /// Regular expression (Rust `regex` crate syntax) matched against each item's
        /// short_name.
        regex: String,
    },
    /// Send a syntactically invalid regex to FindBlockDefs, to validate/demonstrate this
    /// client's handling of a returned (non-panic) RPC error.
    TestInvalidRegex,
    /// Call FindItemDefs, to validate/demonstrate this client's handling of a server-side panic
    /// (FindItemDefs is currently unimplemented on the server).
    TestItemDefsPanic,
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
        Command::FindBlockDefs { regex } => run_find_block_defs(&mut client, regex).await,
        Command::FindItemDefs { regex } => run_find_item_defs(&mut client, regex).await,
        Command::TestInvalidRegex => run_find_block_defs(&mut client, "(".to_string()).await,
        Command::TestItemDefsPanic => run_find_item_defs(&mut client, ".*".to_string()).await,
    }
}

async fn run_find_block_defs(
    client: &mut PerovskiteDebugClient<Channel>,
    regex: String,
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
                println!("---\n{entry:#?}");
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
                println!("---\n{entry:#?}");
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
/// server-side panic (e.g. the FindItemDefs `todo!()` stub), which tonic surfaces the same way.
fn print_rpc_error(status: &Status) {
    println!("ERROR: RPC failed");
    println!("  code: {:?}", status.code());
    println!("  message: {}", status.message());
}
