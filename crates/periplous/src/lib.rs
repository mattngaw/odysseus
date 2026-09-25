//! Training telemetry, recording, metrics, and dashboard support.

pub mod hardware;
#[cfg(any(target_os = "linux", test))]
pub mod server;
