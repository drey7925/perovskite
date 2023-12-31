use std::sync::Arc;

use perovskite_core::chat::ChatMessage;
use perovskite_server::game_state::GameState;
use serde::{Deserialize, Serialize};
use serenity::{
    all::{GatewayIntents, Guild, GuildChannel},
    client::EventHandler,
};

use crate::game_builder::GameBuilder;

#[derive(Clone, Serialize, Deserialize, Debug)]
struct DiscordConfig {
    token: String,
    server_id: u64,
    channel_id: u64,
    nickname: String,
}

pub fn connect(game_builder: &mut GameBuilder) -> anyhow::Result<()> {
    let config_file = game_builder.data_dir().join("discord.ron");
    if !config_file.exists() {
        tracing::info!("No Discord config found; skipping Discord integration");
        return Ok(());
    }

    let config = ron::from_str::<DiscordConfig>(&std::fs::read_to_string(&config_file)?)?;

    game_builder.inner.register_startup_action(move |game| {
        let game = game.clone();
        tokio::spawn(async move {
            let intents = GatewayIntents::GUILD_MESSAGES
                | GatewayIntents::DIRECT_MESSAGES
                | GatewayIntents::MESSAGE_CONTENT;
            let handler = DiscordEventHandler {
                game_state: game.clone(),
                config: config.clone(),
            };
            let mut client = serenity::client::ClientBuilder::new(&config.token, intents)
                .event_handler(handler)
                .await
                .unwrap();
            tracing::info!("Connecting to Discord...");
            client.start().await.unwrap();
            tracing::info!("Connected to Discord");
            game.await_start_shutdown().await;
            tracing::info!("Game shutdown detected; disconnecting from Discord");
            client.shard_manager.shutdown_all().await;
            tracing::info!("Disconnected from Discord");
        });
        Ok(())
    });

    Ok(())
}

struct DiscordEventHandler {
    game_state: Arc<GameState>,
    config: DiscordConfig,
}

#[async_trait::async_trait]
impl EventHandler for DiscordEventHandler {
    async fn ready(&self, ctx: serenity::client::Context, ready: serenity::model::gateway::Ready) {
        for &guild in ready.guilds.iter() {
            let guild_id = guild.id;
            if guild_id.get() != self.config.server_id {
                tracing::info!("Skipping guild with ID {}", guild_id);
                continue;
            }
            tracing::info!("Found guild with ID {}", guild_id);
            guild_id
                .edit_nickname(&ctx, Some(&self.config.nickname))
                .await
                .unwrap();
            let channel = match guild_id
                .channels(&ctx.http)
                .await
                .unwrap()
                .get(&serenity::all::ChannelId::new(self.config.channel_id))
                .cloned()
            {
                Some(channel) => channel,
                None => {
                    tracing::warn!(
                        "No channel with ID {} found in guild {}",
                        self.config.channel_id,
                        guild_id
                    );
                    return;
                }
            };

            tokio::task::spawn(run_outbound_loop(
                ctx.clone(),
                channel,
                self.game_state.clone(),
            ));
        }
    }
    async fn message(
        &self,
        _ctx: serenity::client::Context,
        msg: serenity::model::channel::Message,
    ) {
        if msg.channel_id.get() != self.config.channel_id {
            return;
        }
        if msg.author.bot {
            return;
        }
        self.game_state
            .chat()
            .broadcast_chat_message(ChatMessage::new(
                "[discord]",
                format!("{}: {}", msg.author.name, msg.content),
            ))
            .unwrap();
    }
}

async fn run_outbound_loop(
    ctx: serenity::client::Context,
    channel: GuildChannel,
    game: Arc<GameState>,
) {
    let mut chat_messages = game.chat().subscribe();
    while !game.is_shutting_down() {
        tokio::select! {
            _ = game.await_start_shutdown() => {
                break
            },
            msg = chat_messages.recv() => {
                match msg {
                    Ok(msg) => {
                        if msg.origin() != "[discord]" {
                            channel.say(&ctx, msg.text()).await.unwrap();
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to receive local chat message: {}", e);
                    }
                }
            }
        }
    }
    tracing::info!("Shutting down Discord integration");
}
