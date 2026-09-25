use std::{
    fs, io,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use super::{
    Host, Issue, Snapshot, Unavailable,
    nvidia::Nvidia,
    procfs::{self, CpuSampler},
};

/// A reusable Linux collector. Call `sample` on your sampling thread, not per viewer.
/// The first CPU reading has no utilization rate; a later reading supplies the delta.
#[derive(Default)]
pub struct Collector {
    cpu: CpuSampler,
    nvidia: Nvidia,
}

impl Collector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read host counters and NVIDIA telemetry. Individual source failures become
    /// missing fields plus public error categories. Only an invalid wall clock fails
    /// the whole snapshot. No previous GPU or memory readings are reused on failure.
    pub fn sample(&mut self) -> io::Result<Snapshot> {
        let start = Instant::now();
        let mut issues = Vec::new();
        let counters = read_proc("/proc/stat", "cpu", procfs::cpu_counters, &mut issues);
        let cpu = self.cpu.observe(counters, Instant::now());
        let memory = read_proc("/proc/meminfo", "memory", procfs::memory, &mut issues);
        let gpus = self.nvidia.sample(&mut issues);
        let collected_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| io::Error::other("system clock precedes the Unix epoch"))?
            .as_millis()
            .try_into()
            .map_err(|_| io::Error::other("system clock exceeds timestamp range"))?;
        Ok(Snapshot {
            schema_version: 1,
            collected_at_unix_ms,
            collection_duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            host: Host { cpu, memory },
            gpus,
            issues,
        })
    }
}

fn read_proc<T>(
    path: &str,
    metric: &'static str,
    parse: impl FnOnce(&str) -> Option<T>,
    issues: &mut Vec<Issue>,
) -> Option<T> {
    let reason = match fs::read_to_string(path) {
        Ok(text) => match parse(&text) {
            Some(value) => return Some(value),
            None => Unavailable::InvalidData,
        },
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
            Unavailable::PermissionDenied
        }
        Err(_) => Unavailable::Unavailable,
    };
    issues.push(Issue {
        metric,
        gpu_index: None,
        reason,
    });
    None
}
