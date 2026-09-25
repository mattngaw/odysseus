use std::time::Instant;

use super::{Cpu, Memory};

#[derive(Clone, Debug)]
pub(super) struct CpuCounters {
    ticks: [u64; 8],
    logical_cpus: u32,
}

pub(super) fn cpu_counters(text: &str) -> Option<CpuCounters> {
    let mut lines = text.lines();
    let mut fields = lines.next()?.split_whitespace();
    if fields.next()? != "cpu" {
        return None;
    }
    let mut ticks = [0; 8];
    for value in &mut ticks {
        *value = fields.next()?.parse().ok()?;
    }
    // guest and guest_nice are included in user and nice; do not count them again.
    let logical_cpus = lines
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| {
            name.strip_prefix("cpu")
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))
        })
        .count()
        .try_into()
        .ok()?;
    (logical_cpus > 0).then_some(CpuCounters {
        ticks,
        logical_cpus,
    })
}

#[derive(Default)]
pub(super) struct CpuSampler {
    previous: Option<(CpuCounters, Instant)>,
}

impl CpuSampler {
    pub(super) fn observe(&mut self, current: Option<CpuCounters>, at: Instant) -> Option<Cpu> {
        let Some(current) = current else {
            self.previous = None;
            return None;
        };
        let previous = self.previous.replace((current.clone(), at));
        let measurement = previous.and_then(|(previous, before)| {
            if previous.logical_cpus != current.logical_cpus {
                return None;
            }
            let elapsed = at.checked_duration_since(before)?;
            if elapsed.is_zero() {
                return None;
            }
            let mut delta = [0u64; 8];
            for (i, value) in delta.iter_mut().enumerate() {
                // iowait may decrease. Reset the baseline instead of inventing a rate.
                *value = current.ticks[i].checked_sub(previous.ticks[i])?;
            }
            let total = delta.iter().try_fold(0u64, |sum, n| sum.checked_add(*n))?;
            let idle = delta[3].checked_add(delta[4])?;
            (total > 0).then(|| {
                (
                    100.0 * (total - idle) as f64 / total as f64,
                    elapsed.as_secs_f64() * 1000.0,
                )
            })
        });
        Some(Cpu {
            logical_cpu_count: current.logical_cpus,
            utilization_percent: measurement.map(|m| m.0),
            sample_interval_ms: measurement.map(|m| m.1),
        })
    }
}

pub(super) fn memory(text: &str) -> Option<Memory> {
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let target = match fields.next() {
            Some("MemTotal:") => &mut total,
            Some("MemAvailable:") => &mut available,
            _ => continue,
        };
        if target.is_some() {
            return None;
        }
        let kb: u64 = fields.next()?.parse().ok()?;
        if fields.next()? != "kB" || fields.next().is_some() {
            return None;
        }
        *target = Some(kb.checked_mul(1024)?);
    }
    let total_bytes = total?;
    let available_bytes = available?;
    let used_bytes = total_bytes.checked_sub(available_bytes)?;
    (total_bytes > 0).then_some(Memory {
        total_bytes,
        available_bytes,
        used_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn counters(line: &str) -> Option<CpuCounters> {
        cpu_counters(&format!("{line}\ncpu0 0\ncpu1 0\nintr 999\n"))
    }

    #[test]
    fn cpu_uses_deltas_excludes_iowait_and_does_not_double_count_guests() {
        let mut sampler = CpuSampler::default();
        let at = Instant::now();
        let first = counters("cpu 100 10 20 300 5 2 3 0 70 3");
        assert!(
            sampler
                .observe(first, at)
                .unwrap()
                .utilization_percent
                .is_none()
        );
        // Delta: 10 user + 10 system + 60 idle + 20 iowait = 100, of which 20 busy.
        let next = counters("cpu 110 10 30 360 25 2 3 0 80 3");
        let cpu = sampler.observe(next, at + Duration::from_secs(1)).unwrap();
        assert_eq!(cpu.logical_cpu_count, 2);
        assert_eq!(cpu.utilization_percent, Some(20.0));
        assert_eq!(cpu.sample_interval_ms, Some(1000.0));
    }

    #[test]
    fn discontinuities_and_missing_samples_require_a_new_baseline() {
        let at = Instant::now();
        let normal = counters("cpu 10 0 10 100 5 0 0 0");
        let reset = counters("cpu 20 0 20 200 4 0 0 0");
        let mut sampler = CpuSampler::default();
        sampler.observe(normal.clone(), at);
        assert!(
            sampler
                .observe(normal.clone(), at + Duration::from_secs(1))
                .unwrap()
                .utilization_percent
                .is_none()
        );
        assert!(
            sampler
                .observe(reset.clone(), at + Duration::from_secs(2))
                .unwrap()
                .utilization_percent
                .is_none()
        );
        assert!(sampler.observe(None, at + Duration::from_secs(3)).is_none());
        assert!(
            sampler
                .observe(reset, at + Duration::from_secs(4))
                .unwrap()
                .utilization_percent
                .is_none()
        );
        let mut hotplug = normal.unwrap();
        hotplug.logical_cpus = 3;
        assert!(
            sampler
                .observe(Some(hotplug), at + Duration::from_secs(5))
                .unwrap()
                .utilization_percent
                .is_none()
        );
    }

    #[test]
    fn invalid_cpu_input_and_counter_overflow_do_not_create_fake_rates() {
        assert!(cpu_counters("cpu 1 2\ncpu0 0").is_none());
        assert!(cpu_counters("cpu x 0 0 0 0 0 0 0\ncpu0 0").is_none());
        let at = Instant::now();
        let mut sampler = CpuSampler::default();
        sampler.observe(counters("cpu 0 0 0 0 0 0 0 0"), at);
        let huge = counters(&format!("cpu {} {} 0 0 0 0 0 0", u64::MAX, u64::MAX));
        assert!(
            sampler
                .observe(huge, at + Duration::from_secs(1))
                .unwrap()
                .utilization_percent
                .is_none()
        );
    }

    #[test]
    fn memory_uses_available_instead_of_free_and_converts_kib_to_bytes() {
        assert_eq!(
            memory("MemFree: 10 kB\nMemTotal: 1000 kB\nMemAvailable: 400 kB\n"),
            Some(Memory {
                total_bytes: 1_024_000,
                available_bytes: 409_600,
                used_bytes: 614_400
            })
        );
    }

    #[test]
    fn malformed_or_incomplete_memory_is_unavailable() {
        for text in [
            "MemTotal: 10 kB",
            "MemTotal: 10 MB\nMemAvailable: 5 kB",
            "MemTotal: 10 kB\nMemAvailable: 11 kB",
            "MemTotal: 0 kB\nMemAvailable: 0 kB",
            "MemTotal: 10 kB\nMemTotal: 10 kB\nMemAvailable: 5 kB",
            "MemTotal: 18446744073709551615 kB\nMemAvailable: 0 kB",
        ] {
            assert!(memory(text).is_none(), "{text}");
        }
    }
}
