use std::time::Instant;
// TODO: Conditionally replace with a ZST based on a feature flag
pub struct TraceBuffer {
    created: Instant,
    buf: std::sync::mpsc::SyncSender<(Instant, &'static str)>,
    buf_recv: std::sync::Mutex<std::sync::mpsc::Receiver<(Instant, &'static str)>>,
}

impl TraceBuffer {
    pub fn new() -> TraceBuffer {
        let (tx, rx) = std::sync::mpsc::sync_channel(256);
        TraceBuffer {
            created: Instant::now(),
            buf: tx,
            buf_recv: std::sync::Mutex::new(rx),
        }
    }
    pub fn log(&self, msg: &'static str) {
        let _ = self.buf.try_send((Instant::now(), msg));
    }
}
impl Drop for TraceBuffer {
    fn drop(&mut self) {
        println!("+-----TRACE-----");
        let mut prev_nanos = 0;
        for (i, (when, msg)) in self.buf_recv.lock().unwrap().try_iter().enumerate() {
            let nanos = (when - self.created).as_nanos();
            let diff = nanos - prev_nanos;
            prev_nanos = nanos;
            println!("| {: >4} {: >12} (+{: >12}): {}", i, nanos, diff, msg);
        }
    }
}

pub trait TraceLog {
    fn log(&self, msg: &'static str);
}
impl TraceLog for Option<TraceBuffer> {
    fn log(&self, msg: &'static str) {
        if let Some(buf) = self.as_ref() {
            buf.log(msg);
        }
    }
}
