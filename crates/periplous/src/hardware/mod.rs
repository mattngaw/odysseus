//! Read-only hardware snapshots. Missing readings serialize as `null`, not zero.

#[cfg(target_os = "linux")]
mod collector;
#[cfg(target_os = "linux")]
mod nvidia;
#[cfg(any(target_os = "linux", test))]
mod procfs;
#[cfg(any(target_os = "linux", test))]
mod sanitize;

#[cfg(target_os = "linux")]
pub use collector::Collector;
use serde::Serialize;

/// A collection of sequential readings, not an atomic hardware snapshot.
#[derive(Debug, Serialize)]
pub struct Snapshot {
    pub schema_version: u32,
    /// Wall-clock time at the end of collection, in milliseconds since the Unix epoch.
    pub collected_at_unix_ms: u64,
    /// Monotonic elapsed time spent collecting this snapshot.
    pub collection_duration_ms: f64,
    pub host: Host,
    /// `None` means GPU enumeration failed; an empty list means no GPUs were found.
    pub gpus: Option<Vec<Gpu>>,
    /// Public diagnostics contain fixed labels and error categories, never raw errors.
    pub issues: Vec<Issue>,
}

#[derive(Debug, Default, Serialize)]
pub struct Host {
    pub cpu: Option<Cpu>,
    pub memory: Option<Memory>,
}

#[derive(Debug, Serialize)]
pub struct Cpu {
    pub logical_cpu_count: u32,
    /// Aggregate non-idle time, excluding iowait, as a percentage of all CPU time.
    /// Includes steal time. Guest time is already included in user/nice time.
    /// `None` until two valid samples, and after counter resets or CPU hotplug.
    pub utilization_percent: Option<f64>,
    pub sample_interval_ms: Option<f64>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Memory {
    pub total_bytes: u64,
    pub available_bytes: u64,
    /// MemTotal minus MemAvailable; reclaimable cache is not counted as used.
    pub used_bytes: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct Gpu {
    pub index: u32,
    pub name: Option<String>,
    pub utilization_percent: Option<u32>,
    pub memory: Option<GpuMemory>,
    pub power_watts: Option<f64>,
    pub temperature_celsius: Option<u32>,
    pub sm_clock_mhz: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
    /// NVML's KB/s over its own sampling window, not our CPU sample interval.
    pub pcie_receive_kb_per_second: Option<u32>,
    pub pcie_send_kb_per_second: Option<u32>,
    pub processes: GpuProcesses,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct GpuMemory {
    pub total_bytes: u64,
    pub used_bytes: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct GpuProcesses {
    /// A PID can appear in both groups; do not sum their VRAM without deduplicating.
    /// Each group is `None` if its query failed, or empty if no processes were found.
    pub compute: Option<Vec<GpuProcess>>,
    pub graphics: Option<Vec<GpuProcess>>,
}

#[derive(Debug, Serialize)]
pub struct GpuProcess {
    pub pid: u32,
    /// Sanitized executable basename, falling back to the kernel's short process
    /// name (which can be truncated). `None` if neither can be read. Never argv.
    pub name: Option<String>,
    pub used_memory_bytes: Option<u64>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Issue {
    pub metric: &'static str,
    pub gpu_index: Option<u32>,
    pub reason: Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unavailable {
    NotSupported,
    PermissionDenied,
    InvalidData,
    Unavailable,
}
