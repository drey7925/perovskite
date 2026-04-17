//! Event bus for audio events.
//!
//! Eventually, this may become highly optimized. For now, it's just a thin wrapper around a tokio
//! broadcast.

use anyhow::{bail, Result};
use cgmath::Vector3;
use tokio::sync::broadcast::error::RecvError;

const CROSSBAR_MESSAGE_CAPACITY: usize = 1024;

/// An audio event broadcast over the audio crossbar to all connected clients.
#[derive(Copy, Clone, Debug)]
pub struct AudioEvent {
    // If we're relaying a sound that came from a client's physics, the session ID associated with
    // it. Otherwise, 0.
    //
    // Note that this should only be set for sounds that come from core player physics, and are
    // already playing on the sender's client. e.g. footsteps
    //
    // It should be unset if e.g. game content plays a sound and everyone (including the player
    // that triggered game content) should hear it, e.g. a music block activated by a player pushing
    // a button
    /// If this sound came from a client's physics/footsteps, the context ID of the client.
    ///
    /// If this came from programmatic game content, 0. If you are unsure, you probably want 0.
    ///
    /// Clients with a matching session ID will suppress playback to avoid double-playing footsteps.
    pub initiating_context_id: usize,
    /// The actual audio instruction to carry out on the client.
    pub instruction: AudioInstruction,
}

/// The audio instruction to carry out on clients receiving an [`AudioEvent`].
#[derive(Copy, Clone, Debug)]
pub enum AudioInstruction {
    PlaySampledSound(perovskite_core::protocol::audio::SampledSoundPlayback),
}

/// A handle that allows sending audio events.
pub struct AudioCrossbarSender {
    sender: tokio::sync::broadcast::Sender<AudioEvent>,
}

impl AudioCrossbarSender {
    pub(crate) fn new() -> AudioCrossbarSender {
        AudioCrossbarSender {
            sender: tokio::sync::broadcast::channel(CROSSBAR_MESSAGE_CAPACITY).0,
        }
    }

    /// Broadcasts an audio event to all current subscribers. Errors are silently dropped.
    pub fn send_event(&self, event: AudioEvent) {
        // Drop because we don't care about the result of sending into a broadcast
        let _ = self.sender.send(event);
    }

    /// Creates a new receiver that will receive future audio events.
    ///
    /// This is exposed as public, but probably isn't useful except either in tests or in specialized
    /// mods.
    pub fn subscribe(&self) -> AudioCrossbarReceiver {
        AudioCrossbarReceiver {
            receiver: self.sender.subscribe(),
        }
    }
}

/// Receiving end of the audio crossbar. Obtain one via [`AudioCrossbarSender::subscribe`].
///
/// Note that this type is unstable, and may require additional attention to use in the future
/// (e.g. if spatial optimizations are added it'll be necessary to indicate where the "listener" represented
/// by this receiver is located in space)
pub struct AudioCrossbarReceiver {
    receiver: tokio::sync::broadcast::Receiver<AudioEvent>,
}
impl AudioCrossbarReceiver {
    /// Waits for the next audio event. Automatically recovers from lag by resubscribing.
    pub async fn recv(&mut self, _player_coord: Vector3<f64>) -> Result<AudioEvent> {
        match self.receiver.recv().await {
            Ok(x) => Ok(x),
            Err(RecvError::Lagged(_)) => {
                self.receiver = self.receiver.resubscribe();
                // Try receiving again once. If we get lagged a second time in a row, woe to us.
                let fallback_result = self.receiver.recv().await?;
                Ok(fallback_result)
            }
            Err(RecvError::Closed) => {
                bail!("Channel closed")
            }
        }
    }
}
