use std::fs;

use nvml_wrapper::{
    Nvml,
    enum_wrappers::device::{Clock, PcieUtilCounter, TemperatureSensor},
    enums::device::UsedGpuMemory,
    error::NvmlError,
    struct_wrappers::device::ProcessInfo,
};

use super::{
    Gpu, GpuMemory, GpuProcess, GpuProcesses, Issue, Unavailable,
    sanitize::{executable_name, kernel_process_name},
};

#[derive(Default)]
pub(super) struct Nvidia {
    nvml: Option<Nvml>,
}

impl Nvidia {
    pub(super) fn sample(&mut self, issues: &mut Vec<Issue>) -> Option<Vec<Gpu>> {
        if self.nvml.is_none() {
            self.nvml = reading(Nvml::init(), "nvidia", None, issues);
        }
        let nvml = self.nvml.as_ref()?;
        let count = reading(nvml.device_count(), "gpu_count", None, issues)?;
        Some(
            (0..count)
                .map(|index| {
                    let device =
                        reading(nvml.device_by_index(index), "device", Some(index), issues);
                    let Some(device) = device else {
                        return Gpu {
                            index,
                            ..Gpu::default()
                        };
                    };
                    let mut read = |metric, value| reading(value, metric, Some(index), issues);
                    let utilization_percent = read(
                        "utilization",
                        device.utilization_rates().and_then(|u| {
                            if u.gpu <= 100 {
                                Ok(u.gpu)
                            } else {
                                Err(NvmlError::InvalidArg)
                            }
                        }),
                    );
                    let temperature_celsius =
                        read("temperature", device.temperature(TemperatureSensor::Gpu));
                    let sm_clock_mhz = read("sm_clock", device.clock_info(Clock::SM));
                    let memory_clock_mhz = read("memory_clock", device.clock_info(Clock::Memory));
                    let power_watts =
                        read("power", device.power_usage()).map(|mw| f64::from(mw) / 1000.0);
                    let pcie_receive_kb_per_second = read(
                        "pcie_receive",
                        device.pcie_throughput(PcieUtilCounter::Receive),
                    );
                    let pcie_send_kb_per_second =
                        read("pcie_send", device.pcie_throughput(PcieUtilCounter::Send));
                    let memory = reading(device.memory_info(), "gpu_memory", Some(index), issues)
                        .and_then(|m| {
                            if m.total > 0 && m.used <= m.total {
                                Some(GpuMemory {
                                    total_bytes: m.total,
                                    used_bytes: m.used,
                                })
                            } else {
                                issues.push(Issue {
                                    metric: "gpu_memory",
                                    gpu_index: Some(index),
                                    reason: Unavailable::InvalidData,
                                });
                                None
                            }
                        });
                    Gpu {
                        index,
                        name: reading(device.name(), "gpu_name", Some(index), issues),
                        utilization_percent,
                        memory,
                        power_watts,
                        temperature_celsius,
                        sm_clock_mhz,
                        memory_clock_mhz,
                        pcie_receive_kb_per_second,
                        pcie_send_kb_per_second,
                        processes: GpuProcesses {
                            compute: reading(
                                device.running_compute_processes(),
                                "compute_processes",
                                Some(index),
                                issues,
                            )
                            .map(processes),
                            graphics: reading(
                                device.running_graphics_processes(),
                                "graphics_processes",
                                Some(index),
                                issues,
                            )
                            .map(processes),
                        },
                    }
                })
                .collect(),
        )
    }
}

fn processes(rows: Vec<ProcessInfo>) -> Vec<GpuProcess> {
    let mut processes: Vec<_> = rows
        .into_iter()
        .map(|p| {
            // Read only exe/comm, never cmdline, environ, or owner details.
            // A process can exit or become inaccessible between the NVML query and this read.
            let name = fs::read_link(format!("/proc/{}/exe", p.pid))
                .ok()
                .and_then(|path| executable_name(&path))
                .or_else(|| {
                    fs::read_to_string(format!("/proc/{}/comm", p.pid))
                        .ok()
                        .and_then(|comm| kernel_process_name(&comm))
                });
            GpuProcess {
                pid: p.pid,
                name,
                used_memory_bytes: match p.used_gpu_memory {
                    UsedGpuMemory::Unavailable => None,
                    UsedGpuMemory::Used(bytes) => Some(bytes),
                },
            }
        })
        .collect();
    processes.sort_by_key(|p| p.pid);
    processes
}

fn reading<T>(
    result: Result<T, NvmlError>,
    metric: &'static str,
    gpu_index: Option<u32>,
    issues: &mut Vec<Issue>,
) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            let reason = match error {
                NvmlError::NotSupported
                | NvmlError::FunctionNotFound
                | NvmlError::FailedToLoadSymbol(_) => Unavailable::NotSupported,
                NvmlError::NoPermission => Unavailable::PermissionDenied,
                NvmlError::InvalidArg | NvmlError::Utf8Error(_) => Unavailable::InvalidData,
                _ => Unavailable::Unavailable,
            };
            issues.push(Issue {
                metric,
                gpu_index,
                reason,
            });
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_support_is_distinct_from_zero_and_raw_errors_are_not_published() {
        let mut issues = vec![];
        assert_eq!(reading(Ok(0), "power", Some(0), &mut issues), Some(0));
        assert!(
            reading::<u32>(
                Err(NvmlError::FailedToLoadSymbol("/private/path/secret".into())),
                "power",
                Some(0),
                &mut issues
            )
            .is_none()
        );
        let json = serde_json::to_string(&issues).unwrap();
        assert!(json.contains("not_supported"));
        assert!(!json.contains("private"));
        assert!(!json.contains("secret"));
    }

    #[test]
    fn missing_process_and_unavailable_memory_are_not_reported_as_zero() {
        let rows = vec![ProcessInfo {
            pid: u32::MAX,
            used_gpu_memory: UsedGpuMemory::Unavailable,
            gpu_instance_id: None,
            compute_instance_id: None,
        }];
        let values = processes(rows);
        assert!(values[0].name.is_none());
        assert!(values[0].used_memory_bytes.is_none());
    }
}
