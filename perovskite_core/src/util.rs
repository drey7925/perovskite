use std::{
    fmt::Debug,
    sync::{atomic::AtomicUsize, Arc},
    time::Instant,
};

use rand::Rng;
// TODO: Conditionally replace with a ZST based on a feature flag

static TRACE_RATE_DENOMINATOR: AtomicUsize = AtomicUsize::new(1);

pub fn set_trace_rate_denominator(val: usize) {
    TRACE_RATE_DENOMINATOR.store(val, std::sync::atomic::Ordering::Relaxed);
}

struct TraceBufferInner {
    created: Instant,
    buf: std::sync::mpsc::SyncSender<(Instant, &'static str)>,
    buf_recv: std::sync::Mutex<std::sync::mpsc::Receiver<(Instant, &'static str)>>,
}

pub struct TraceBuffer {
    inner: Option<Arc<TraceBufferInner>>,
}
impl Clone for TraceBuffer {
    fn clone(&self) -> Self {
        self.log("Cloning trace buffer");
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl Debug for TraceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.inner.is_some() {
            f.write_str("TraceBuffer")
        } else {
            f.write_str("Empty TraceBuffer")
        }
    }
}

impl TraceBuffer {
    pub fn new(force_print: bool) -> TraceBuffer {
        if force_print
            || rand::thread_rng().gen_bool(
                1.0 / TRACE_RATE_DENOMINATOR.load(std::sync::atomic::Ordering::Relaxed) as f64,
            )
        {
            Self::new_filled()
        } else {
            Self::empty()
        }
    }
    pub fn log(&self, msg: &'static str) {
        if let Some(inner) = self.inner.as_ref() {
            let _ = inner.buf.try_send((Instant::now(), msg));
        }
    }
    fn new_filled() -> TraceBuffer {
        let (tx, rx) = std::sync::mpsc::sync_channel(256);
        let inner = TraceBufferInner {
            created: Instant::now(),
            buf: tx,
            buf_recv: std::sync::Mutex::new(rx),
        };
        TraceBuffer {
            inner: Some(Arc::new(inner)),
        }
    }
    pub fn empty() -> TraceBuffer {
        TraceBuffer { inner: None }
    }
}
impl Drop for TraceBufferInner {
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
