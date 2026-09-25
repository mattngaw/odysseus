use super::*;
use crate::hardware::Host;
use axum::body::to_bytes;
use tower::ServiceExt;

fn hosts() -> AllowedHosts {
    AllowedHosts::new("localhost,127.0.0.1").unwrap()
}

fn snapshot() -> Snapshot {
    Snapshot {
        schema_version: 1,
        collected_at_unix_ms: 0,
        collection_duration_ms: 1.0,
        host: Host::default(),
        gpus: None,
        issues: vec![],
    }
}

#[test]
fn only_fixed_read_only_routes_are_available() {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let app = router(Cache::new(), Deployment::default(), hosts());
        for path in [
            "/",
            "/style.css",
            "/app.js",
            "/view-model.js",
            "/api/hardware",
            "/api/deployment",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .header(header::HOST, "localhost")
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(response.headers()["content-security-policy"], CSP);
            assert_eq!(response.headers()["cache-control"], "no-store");
            assert!(
                !response
                    .headers()
                    .contains_key("access-control-allow-origin")
            );
        }
        for path in [
            "/etc/passwd",
            "/../Cargo.toml",
            "/%2e%2e/Cargo.toml",
            "/api/exec",
            "/api/hardware/file",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .header(header::HOST, "localhost")
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }
        for method in [
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .header(header::HOST, "localhost")
                        .method(method)
                        .uri("/api/hardware")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
            assert_eq!(response.headers()[header::ALLOW], "GET, HEAD");
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .header(header::HOST, "localhost")
                    .uri("/api/hardware")
                    .header(header::CONTENT_LENGTH, "4")
                    .body(Body::from("test"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        for path in ["/", "/api/hardware", "/missing"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .header(header::HOST, "localhost")
                        .method(Method::HEAD)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert!(to_bytes(response.into_body(), 1).await.unwrap().is_empty());
        }
    });
}

#[test]
fn readers_share_a_snapshot_and_staleness_is_explicit() {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let cache = Cache::new();
        let app = router(cache.clone(), Deployment::default(), hosts());
        let request = || {
            Request::builder()
                .header(header::HOST, "localhost")
                .uri("/api/hardware")
                .body(Body::empty())
                .unwrap()
        };
        let initial = app.clone().oneshot(request()).await.unwrap();
        assert_eq!(initial.headers()["x-periplous-stale"], "true");
        let body = to_bytes(initial.into_body(), usize::MAX).await.unwrap();
        assert!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["snapshot"].is_null());

        let json = History::default().encode(&snapshot(), 0);
        cache.publish(json.clone(), Instant::now());
        for _ in 0..3 {
            let response = app.clone().oneshot(request()).await.unwrap();
            assert_eq!(response.headers()["x-periplous-stale"], "false");
            assert_eq!(
                to_bytes(response.into_body(), usize::MAX).await.unwrap(),
                json
            );
        }
        cache.publish(json.clone(), Instant::now() - Duration::from_secs(10));
        let response = app.oneshot(request()).await.unwrap();
        assert_eq!(response.headers()["x-periplous-stale"], "true");
        assert!(
            response.headers()["x-periplous-sample-age-ms"]
                .to_str()
                .unwrap()
                .parse::<u64>()
                .unwrap()
                >= 10_000
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await.unwrap(),
            json
        );
    });
}

#[test]
fn history_is_bounded_by_count_and_time_and_preserves_unavailability() {
    let mut history = History::default();
    let sample = snapshot();
    for tick in 0..620 {
        history.encode(&sample, tick * 1_000);
    }
    assert_eq!(history.points.len(), 600);
    assert_eq!(history.points.front().unwrap().elapsed_ms, 20_000);
    assert_eq!(history.sequence, 620);
    let json: serde_json::Value =
        serde_json::from_slice(&history.encode(&sample, 1_300_000)).unwrap();
    assert_eq!(json["history"].as_array().unwrap().len(), 1);
    assert!(json["history"][0]["gpus"].is_null());
    assert_eq!(json["sequence"], 621);
}

#[test]
fn deployment_exposes_only_valid_public_identifiers() {
    let invalid = Deployment::validated(Some("/private/config".into()), Some("secret".into()));
    assert!(invalid.environment.is_none());
    assert!(invalid.release.is_none());
    let release = format!("r-{}", "a".repeat(64));
    let deployment = Deployment::validated(Some("dev".into()), Some(release.clone()));
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let response = router(Cache::new(), deployment, hosts())
            .oneshot(
                Request::builder()
                    .header(header::HOST, "localhost")
                    .uri("/api/deployment")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"environment": "dev", "release": release})
        );
    });
}

#[test]
fn unknown_duplicate_and_mismatched_hosts_are_rejected() {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let app = router(Cache::new(), Deployment::default(), hosts());
        for request in [
            Request::builder()
                .uri("/api/hardware")
                .body(Body::empty())
                .unwrap(),
            Request::builder()
                .uri("/api/hardware")
                .header(header::HOST, "attacker.example:8765")
                .body(Body::empty())
                .unwrap(),
            Request::builder()
                .uri("/api/hardware")
                .header(header::HOST, "localhost")
                .header(header::HOST, "attacker.example")
                .body(Body::empty())
                .unwrap(),
            Request::builder()
                .uri("http://attacker.example/api/hardware")
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        ] {
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::MISDIRECTED_REQUEST);
            assert_eq!(response.headers()["cache-control"], "no-store");
        }
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/hardware")
                    .header(header::HOST, "LOCALHOST:8765")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    });
}

#[test]
fn transport_bounds_incomplete_connections_and_releases_capacity() {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        time::{sleep, timeout},
    };
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let limits = transport::Limits {
            connections: 1,
            header_timeout: Duration::from_millis(150),
            connection_timeout: Duration::from_millis(500),
        };
        let server = tokio::spawn(transport::serve(
            listener,
            router(Cache::new(), Deployment::default(), hosts()),
            limits,
        ));
        let mut slow = TcpStream::connect(address).await.unwrap();
        slow.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nX-Slow: ")
            .await
            .unwrap();
        // Give the accept loop time to allocate the first permit before the second client.
        sleep(Duration::from_millis(30)).await;
        let mut overflow = TcpStream::connect(address).await.unwrap();
        let mut byte = [0; 1];
        let closed = timeout(Duration::from_millis(100), overflow.read(&mut byte))
            .await
            .unwrap();
        assert!(
            matches!(closed, Ok(0) | Err(_)),
            "excess connection must close without waiting for capacity"
        );
        let mut response = Vec::new();
        timeout(Duration::from_millis(400), slow.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(!response.starts_with(b"HTTP/1.1 200"));
        assert!(response.is_empty() || response.starts_with(b"HTTP/1.1 408"));
        let mut healthy = TcpStream::connect(address).await.unwrap();
        healthy
            .write_all(b"GET /api/deployment HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        response.clear();
        timeout(
            Duration::from_millis(400),
            healthy.read_to_end(&mut response),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(
            String::from_utf8(response)
                .unwrap()
                .to_ascii_lowercase()
                .contains("connection: close")
        );
        server.abort();
    });
}

#[test]
fn absolute_connection_deadline_bounds_a_stalled_response() {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        time::timeout,
    };
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/",
            get(|| async { std::future::pending::<&'static str>().await }),
        );
        let server = tokio::spawn(transport::serve(
            listener,
            app,
            transport::Limits {
                connections: 1,
                header_timeout: Duration::from_millis(50),
                connection_timeout: Duration::from_millis(150),
            },
        ));
        let mut client = TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        timeout(
            Duration::from_millis(500),
            client.read_to_end(&mut response),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(response.is_empty());
        server.abort();
    });
}
