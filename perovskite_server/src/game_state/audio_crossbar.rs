//! Event bus for audio events.
//!
//! Eventually, this may become highly optimized. For now, it's just a thin wrapper around a tokio
//! broadcast.

use anyhow::{bail, Result};
use cgmath::Vector3;
use tokio::sync::broadcast::error::RecvError;

const CROSSBAR_MESSAGE_CAPACITY: usize = 1024;

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
    pub initiating_context_id: usize,
    pub instruction: AudioInstruction,
}

#[derive(Copy, Clone, Debug)]
pub enum AudioInstruction {
    PlaySampledSound(perovskite_core::protocol::audio::SampledSoundPlayback),
}

pub struct AudioCrossbarSender {
    sender: tokio::sync::broadcast::Sender<AudioEvent>,
}

impl AudioCrossbarSender {
    pub(crate) fn new() -> AudioCrossbarSender {
        AudioCrossbarSender {
            sender: tokio::sync::broadcast::channel(CROSSBAR_MESSAGE_CAPACITY).0,
        }
    }

    pub fn send_event(&self, event: AudioEvent) {
        // Drop because we don't care about the result of sending into a broadcast
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> AudioCrossbarReceiver {
        AudioCrossbarReceiver {
            receiver: self.sender.subscribe(),
        }
    }
}

pub struct AudioCrossbarReceiver {
    receiver: tokio::sync::broadcast::Receiver<AudioEvent>,
}
impl AudioCrossbarReceiver {
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
