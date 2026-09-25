use std::{
    env,
    ffi::OsString,
    net::{Ipv4Addr, SocketAddrV4},
    process::ExitCode,
};

const HELP: &str = "Usage: periplous snapshot\n       periplous serve [--bind IPV4] [--port PORT]\n       periplous serve --systemd\n\nsnapshot: print a read-only Linux hardware snapshot as JSON.\n          Samples CPU counters across one second; unavailable readings are null.\nserve:    sample once per second and serve the dashboard.\n          Defaults to 127.0.0.1:8765; --bind 0.0.0.0 listens on all IPv4 interfaces.\n";

fn main() -> ExitCode {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    match arguments.as_slice() {
        [command] if command == "--help" || command == "-h" => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        [command] if command == "snapshot" => finish(snapshot()),
        [command, option] if command == "serve" && option == "--systemd" => finish(serve(None)),
        [command, options @ ..] if command == "serve" => match parse_address(options) {
            Ok(address) => finish(serve(Some(address))),
            Err(message) => {
                eprintln!("{message}");
                ExitCode::from(2)
            }
        },
        _ => {
            eprint!("{HELP}");
            ExitCode::from(2)
        }
    }
}

fn parse_address(options: &[OsString]) -> Result<SocketAddrV4, &'static str> {
    let mut bind = None;
    let mut port = None;
    let mut pairs = options.chunks_exact(2);
    for pair in &mut pairs {
        match pair[0].to_str() {
            Some("--bind") if bind.is_none() => {
                bind = Some(
                    pair[1]
                        .to_str()
                        .and_then(|s| s.parse::<Ipv4Addr>().ok())
                        .ok_or("bind must be an IPv4 address")?,
                );
            }
            Some("--port") if port.is_none() => {
                port = Some(
                    pair[1]
                        .to_str()
                        .and_then(|s| s.parse::<u16>().ok())
                        .filter(|p| *p > 0)
                        .ok_or("port must be between 1 and 65535")?,
                );
            }
            _ => return Err("unknown or repeated serve option"),
        }
    }
    if !pairs.remainder().is_empty() {
        return Err("each serve option requires a value");
    }
    Ok(SocketAddrV4::new(
        bind.unwrap_or(Ipv4Addr::LOCALHOST),
        port.unwrap_or(8765),
    ))
}

fn finish(result: std::io::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("periplous: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "linux")]
fn serve(address: Option<SocketAddrV4>) -> std::io::Result<()> {
    periplous::server::run(address)
}

#[cfg(not(target_os = "linux"))]
fn serve(_address: Option<SocketAddrV4>) -> std::io::Result<()> {
    snapshot()
}

#[cfg(target_os = "linux")]
fn snapshot() -> std::io::Result<()> {
    use periplous::hardware::Collector;
    use std::{
        io::{self, Write},
        thread,
        time::Duration,
    };

    let mut collector = Collector::new();
    collector.sample()?;
    thread::sleep(Duration::from_secs(1));
    let snapshot = collector.sample()?;
    let mut output = io::stdout().lock();
    serde_json::to_writer_pretty(&mut output, &snapshot)?;
    writeln!(output)
}

#[cfg(not(target_os = "linux"))]
fn snapshot() -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "hardware collection currently requires Linux",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address(options: &[&str]) -> Result<SocketAddrV4, &'static str> {
        parse_address(&options.iter().map(OsString::from).collect::<Vec<_>>())
    }

    #[test]
    fn serving_defaults_to_loopback_and_requires_an_explicit_wildcard() {
        assert_eq!(address(&[]).unwrap().to_string(), "127.0.0.1:8765");
        assert_eq!(
            address(&["--bind", "0.0.0.0"]).unwrap().to_string(),
            "0.0.0.0:8765"
        );
        for options in [
            ["--port", "9000", "--bind", "198.51.100.32"],
            ["--bind", "198.51.100.32", "--port", "9000"],
        ] {
            assert_eq!(address(&options).unwrap().to_string(), "198.51.100.32:9000");
        }
    }

    #[test]
    fn invalid_or_ambiguous_addresses_are_rejected_before_startup() {
        for options in [
            vec!["--bind"],
            vec!["--bind", "hostname"],
            vec!["--bind", "::"],
            vec!["--bind", "256.0.0.1"],
            vec!["--port", "0"],
            vec!["--port", "65536"],
            vec!["--port", "8080", "--port", "8765"],
            vec!["--bind", "127.0.0.1", "--bind", "0.0.0.0"],
            vec!["--unknown", "value"],
        ] {
            assert!(address(&options).is_err(), "{options:?}");
        }
    }
}
