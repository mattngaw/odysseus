//! Read-only HTTP dashboard. No route reads files or collects hardware.

use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{Request, State},
    http::{HeaderValue, Method, StatusCode, header, uri::Authority},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Serialize;

use crate::hardware::Snapshot;

mod transport;

const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const HISTORY_WINDOW_MS: u64 = 600_000;
const HISTORY_CAPACITY: usize = 600;
const STALE_AFTER: Duration = Duration::from_secs(5);
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

#[derive(Clone)]
struct Cache(Arc<RwLock<Published>>);

struct Published {
    json: Bytes,
    updated: Option<Instant>,
}

impl Cache {
    fn new() -> Self {
        Self(Arc::new(RwLock::new(Published {
            json: Bytes::from_static(b"{\"sample_interval_ms\":1000,\"history_window_ms\":600000,\"sequence\":0,\"snapshot\":null,\"history\":[]}"),
            updated: None,
        })))
    }

    fn publish(&self, json: Vec<u8>, updated: Instant) {
        *self.0.write().expect("telemetry cache lock poisoned") = Published {
            json: json.into(),
            updated: Some(updated),
        };
    }
}

#[derive(Serialize)]
struct HistoryGpu {
    index: u32,
    utilization_percent: Option<u32>,
    memory_percent: Option<f64>,
}

#[derive(Serialize)]
struct HistoryPoint {
    /// Monotonic offset from server startup; unaffected by wall-clock adjustments.
    elapsed_ms: u64,
    gpus: Option<Vec<HistoryGpu>>,
}

#[derive(Default)]
struct History {
    points: VecDeque<HistoryPoint>,
    sequence: u64,
}

impl History {
    fn encode(&mut self, snapshot: &Snapshot, elapsed_ms: u64) -> Vec<u8> {
        self.points.push_back(HistoryPoint {
            elapsed_ms,
            gpus: snapshot.gpus.as_ref().map(|gpus| {
                gpus.iter()
                    .map(|g| HistoryGpu {
                        index: g.index,
                        utilization_percent: g.utilization_percent,
                        memory_percent: g
                            .memory
                            .as_ref()
                            .filter(|m| m.total_bytes > 0)
                            .map(|m| 100.0 * m.used_bytes as f64 / m.total_bytes as f64),
                    })
                    .collect()
            }),
        });
        while self.points.len() > HISTORY_CAPACITY
            || self
                .points
                .front()
                .is_some_and(|p| elapsed_ms.saturating_sub(p.elapsed_ms) > HISTORY_WINDOW_MS)
        {
            self.points.pop_front();
        }
        self.sequence += 1;
        #[derive(Serialize)]
        struct Document<'a> {
            sample_interval_ms: u64,
            history_window_ms: u64,
            sequence: u64,
            snapshot: &'a Snapshot,
            history: &'a VecDeque<HistoryPoint>,
        }
        serde_json::to_vec(&Document {
            sample_interval_ms: SAMPLE_INTERVAL.as_millis() as u64,
            history_window_ms: HISTORY_WINDOW_MS,
            sequence: self.sequence,
            snapshot,
            history: &self.points,
        })
        .expect("snapshot serialization cannot fail")
    }
}

/// Only deliberately public identifiers, never arbitrary environment contents.
#[derive(Clone, Default, Serialize)]
struct Deployment {
    environment: Option<String>,
    release: Option<String>,
}

#[cfg(any(target_os = "linux", test))]
impl Deployment {
    fn validated(environment: Option<String>, release: Option<String>) -> Self {
        Self {
            environment: environment.filter(|s| s == "dev" || s == "prod"),
            release: release.filter(|s| {
                s.len() == 66
                    && s.starts_with("r-")
                    && s.as_bytes()[2..]
                        .iter()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
            }),
        }
    }
}

#[derive(Clone)]
struct AllowedHosts(Arc<HashSet<String>>);

impl AllowedHosts {
    fn new(hosts: &str) -> std::io::Result<Self> {
        let mut allowed = HashSet::new();
        for host in hosts.split(',') {
            let host = host.trim().to_ascii_lowercase();
            let authority = host
                .parse::<Authority>()
                .map_err(|_| std::io::Error::other("invalid allowed host"))?;
            if host.is_empty() || host.contains(['@', '*']) || authority.port().is_some() {
                return Err(std::io::Error::other(
                    "allowed hosts must be names or IPs without ports",
                ));
            }
            allowed.insert(host);
        }
        Ok(Self(Arc::new(allowed)))
    }

    fn accepts(&self, request: &Request) -> bool {
        let mut hosts = request.headers().get_all(header::HOST).iter();
        let Some(host) = hosts.next().and_then(|h| h.to_str().ok()) else {
            return false;
        };
        if hosts.next().is_some() || host.contains('@') {
            return false;
        }
        let Ok(authority) = host.parse::<Authority>() else {
            return false;
        };
        self.0.contains(&authority.host().to_ascii_lowercase())
            && request.uri().authority().is_none_or(|a| a == &authority)
    }
}

fn router(cache: Cache, deployment: Deployment, hosts: AllowedHosts) -> Router {
    let deployment =
        Bytes::from(serde_json::to_vec(&deployment).expect("deployment serialization"));
    Router::new()
        .route(
            "/",
            get(|| async {
                asset(
                    "text/html; charset=utf-8",
                    include_str!("../web/index.html"),
                )
            }),
        )
        .route(
            "/style.css",
            get(|| async { asset("text/css; charset=utf-8", include_str!("../web/style.css")) }),
        )
        .route(
            "/app.js",
            get(|| async {
                asset(
                    "text/javascript; charset=utf-8",
                    include_str!("../web/app.js"),
                )
            }),
        )
        .route(
            "/view-model.js",
            get(|| async {
                asset(
                    "text/javascript; charset=utf-8",
                    include_str!("../web/view-model.js"),
                )
            }),
        )
        .route("/favicon.ico", get(|| async { StatusCode::NO_CONTENT }))
        .route("/api/hardware", get(hardware))
        .route(
            "/api/deployment",
            get(move || {
                let deployment = deployment.clone();
                async move { ([(header::CONTENT_TYPE, "application/json")], deployment) }
            }),
        )
        .fallback(|| async { (StatusCode::NOT_FOUND, "Not found\n") })
        .layer(middleware::from_fn_with_state(hosts, read_only))
        .with_state(cache)
}

fn asset(content_type: &'static str, body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, content_type)], body).into_response()
}

async fn hardware(State(cache): State<Cache>) -> Response {
    let published = cache.0.read().expect("telemetry cache lock poisoned");
    let age = published.updated.map(|t| t.elapsed());
    let mut response = Response::new(Body::from(published.json.clone()));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response.headers_mut().insert(
        "x-periplous-stale",
        HeaderValue::from_static(if age.is_none_or(|d| d > STALE_AFTER) {
            "true"
        } else {
            "false"
        }),
    );
    if let Some(age) = age {
        response.headers_mut().insert(
            "x-periplous-sample-age-ms",
            HeaderValue::from_str(&age.as_millis().to_string()).expect("integer header"),
        );
    }
    response
}

async fn read_only(State(hosts): State<AllowedHosts>, request: Request, next: Next) -> Response {
    let head = request.method() == Method::HEAD;
    let mut response = if !hosts.accepts(&request) {
        (StatusCode::MISDIRECTED_REQUEST, "Unrecognized host\n").into_response()
    } else if request.method() != Method::GET && request.method() != Method::HEAD {
        (
            StatusCode::METHOD_NOT_ALLOWED,
            [(header::ALLOW, "GET, HEAD")],
            "Read-only endpoint\n",
        )
            .into_response()
    } else if request.headers().contains_key(header::TRANSFER_ENCODING)
        || request
            .headers()
            .get(header::CONTENT_LENGTH)
            .is_some_and(|v| v != "0")
    {
        (
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request bodies are not accepted\n",
        )
            .into_response()
    } else {
        next.run(request).await
    };
    for (name, value) in [
        ("content-security-policy", CSP),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        ("cache-control", "no-store"),
        ("x-frame-options", "DENY"),
    ] {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_static(value));
    }
    if head {
        *response.body_mut() = Body::empty();
    }
    response
}

/// Serve the dashboard at the supplied IPv4 address. The one sampler is independent of
/// request traffic; a stalled collector leaves a visibly stale cached response.
#[cfg(target_os = "linux")]
pub fn run(address: Option<std::net::SocketAddrV4>) -> std::io::Result<()> {
    use crate::hardware::Collector;
    use std::thread;

    let defaults = match address {
        Some(address) if !address.ip().is_unspecified() => {
            format!("localhost,127.0.0.1,{}", address.ip())
        }
        _ => "localhost,127.0.0.1".into(),
    };
    let hosts = AllowedHosts::new(&std::env::var("PERIPLOUS_ALLOWED_HOSTS").unwrap_or(defaults))?;
    let inherited = if address.is_none() {
        Some(transport::systemd_listener()?)
    } else {
        None
    };
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(async {
            let listener = match inherited {
                Some(listener) => tokio::net::TcpListener::from_std(listener)?,
                None => {
                    tokio::net::TcpListener::bind(address.expect("explicit listen address")).await?
                }
            };
            let cache = Cache::new();
            let sampler_cache = cache.clone();
            thread::Builder::new()
                .name("periplous-sampler".into())
                .spawn(move || {
                    let origin = Instant::now();
                    let mut collector = Collector::new();
                    let mut history = History::default();
                    loop {
                        let started = Instant::now();
                        if let Ok(snapshot) = collector.sample() {
                            let json =
                                history.encode(&snapshot, origin.elapsed().as_millis() as u64);
                            sampler_cache.publish(json, Instant::now());
                        }
                        thread::sleep(SAMPLE_INTERVAL.saturating_sub(started.elapsed()));
                    }
                })?;
            eprintln!("Periplous listening on http://{}", listener.local_addr()?);
            let deployment = Deployment::validated(
                std::env::var("PERIPLOUS_ENVIRONMENT").ok(),
                std::env::var("PERIPLOUS_RELEASE").ok(),
            );
            transport::serve(
                listener,
                router(cache, deployment, hosts),
                transport::Limits::default(),
            )
            .await
        })
}

#[cfg(test)]
mod tests;
