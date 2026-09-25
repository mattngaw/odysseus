//! Connection limits apply before HTTP parsing, including clients that never finish headers.

use std::{io, sync::Arc, time::Duration};

use axum::Router;
use hyper::server::conn::http1;
use hyper_util::{
    rt::{TokioIo, TokioTimer},
    service::TowerToHyperService,
};
use tokio::{net::TcpListener, sync::Semaphore, time::timeout};

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub connections: usize,
    pub header_timeout: Duration,
    pub connection_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            connections: 64,
            header_timeout: Duration::from_secs(5),
            connection_timeout: Duration::from_secs(10),
        }
    }
}

pub(super) async fn serve(listener: TcpListener, app: Router, limits: Limits) -> io::Result<()> {
    let permits = Arc::new(Semaphore::new(limits.connections));
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            // A peer can abort before accept; resource pressure can also be
            // temporary. Neither should tear down the server or spin in a loop.
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let Ok(permit) = permits.clone().try_acquire_owned() else {
            // Do not queue unbounded sockets or HTTP tasks behind the semaphore.
            drop(stream);
            continue;
        };
        let service = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            let _permit = permit;
            let mut builder = http1::Builder::new();
            builder
                .timer(TokioTimer::new())
                .header_read_timeout(limits.header_timeout)
                .keep_alive(false)
                .max_headers(32)
                .max_buf_size(16 * 1024);
            // One request per connection. The absolute deadline also bounds slow
            // response readers, including cached history retained by a response.
            let _ = timeout(
                limits.connection_timeout,
                builder.serve_connection(TokioIo::new(stream), service),
            )
            .await;
        });
    }
}

/// Consume exactly the one named TCP socket supplied by systemd, without rebinding it.
#[cfg(target_os = "linux")]
pub(super) fn systemd_listener() -> io::Result<std::net::TcpListener> {
    use std::os::fd::FromRawFd;
    if std::env::var("LISTEN_PID").ok().as_deref() != Some(&std::process::id().to_string())
        || std::env::var("LISTEN_FDS").ok().as_deref() != Some("1")
        || std::env::var("LISTEN_FDNAMES").ok().as_deref() != Some("http")
    {
        return Err(io::Error::other(
            "expected one systemd socket named http for this process",
        ));
    }
    // Validate that fd 3 is open before taking ownership. Environment variables
    // alone must not make an invalid descriptor satisfy FromRawFd's contract.
    // SAFETY: fcntl accepts any integer descriptor and reports EBADF for a closed one.
    let flags = unsafe { libc::fcntl(3, libc::F_GETFD) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd 3 is valid and this startup path owns it exclusively.
    if unsafe { libc::fcntl(3, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd 3 was verified open above and is consumed only once at startup.
    let listener = unsafe { std::net::TcpListener::from_raw_fd(3) };
    listener.local_addr()?;
    listener.set_nonblocking(true)?;
    Ok(listener)
}
