use std::{str::SplitWhitespace, time::Duration};

use penteconter::Color;

#[derive(Debug)]
pub(super) struct Limits {
    pub nodes: Option<u64>,
    pub time: Option<Duration>,
    pub wait_for_stop: bool,
    pub searchmoves: Vec<String>,
}

impl Limits {
    pub fn parse(words: SplitWhitespace<'_>, side: Color) -> Result<Self, String> {
        let mut words = words.peekable();
        let (mut nodes, mut movetime, mut wtime, mut btime) = (None, None, None, None);
        let (mut winc, mut binc, mut movestogo) = (0u64, 0u64, 30u64);
        let mut infinite = false;
        let mut searchmoves = Vec::new();
        while let Some(word) = words.next() {
            match word {
                "infinite" => infinite = true,
                "searchmoves" => {
                    let before = searchmoves.len();
                    while words.peek().is_some_and(|word| !is_keyword(word)) {
                        searchmoves.push(words.next().unwrap().to_owned());
                    }
                    if searchmoves.len() == before {
                        return Err("searchmoves requires at least one move".into());
                    }
                }
                "nodes" | "movetime" | "wtime" | "btime" | "winc" | "binc" | "movestogo" => {
                    let number = words
                        .next()
                        .ok_or_else(|| format!("missing {word} value"))?;
                    if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
                        return Err(format!("invalid {word} value: {number}"));
                    }
                    let number: u64 = number
                        .parse()
                        .map_err(|_| format!("invalid {word} value: {number}"))?;
                    match word {
                        "nodes" => nodes = Some(number),
                        "movetime" => movetime = Some(number),
                        "wtime" => wtime = Some(number),
                        "btime" => btime = Some(number),
                        "winc" => winc = number,
                        "binc" => binc = number,
                        "movestogo" if number != 0 => movestogo = number,
                        _ => return Err("movestogo must be positive".into()),
                    }
                }
                "depth" | "mate" | "ponder" => {
                    return Err(format!("{word} search is not supported"));
                }
                _ => {} // UCI permits ignoring unrecognized tokens.
            }
        }
        let (remaining, increment) = match side {
            Color::White => (wtime, winc),
            Color::Black => (btime, binc),
        };
        let time = movetime
            .or_else(|| {
                remaining.map(|remaining| {
                    // Deliberately simple time allocation, reserving clock for transport
                    // and avoiding spending an increment before it is received.
                    let reserve = (remaining / 20).clamp(10, 1000);
                    (remaining / movestogo)
                        .saturating_add(increment.saturating_mul(3) / 4)
                        .min(remaining.saturating_sub(reserve))
                })
            })
            .map(Duration::from_millis);
        if !infinite
            && movetime.is_none()
            && remaining.is_none()
            && (wtime.is_some() || btime.is_some())
        {
            return Err("missing clock for the side to move".into());
        }
        Ok(Self {
            nodes: if infinite { None } else { nodes },
            time: if infinite { None } else { time },
            wait_for_stop: infinite || (nodes.is_none() && time.is_none()),
            searchmoves,
        })
    }
}

fn is_keyword(word: &str) -> bool {
    matches!(
        word,
        "infinite"
            | "searchmoves"
            | "nodes"
            | "movetime"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "depth"
            | "mate"
            | "ponder"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_support_nodes_time_clocks_infinite_and_zero_work() {
        let parse = |s: &str, side| Limits::parse(s.split_whitespace(), side).unwrap();
        let limits = parse("nodes 0 movetime 50", Color::White);
        assert_eq!(limits.nodes, Some(0));
        assert_eq!(limits.time, Some(Duration::from_millis(50)));
        assert!(!limits.wait_for_stop);
        let limits = parse(
            "wtime 30000 btime 60000 winc 1000 binc 2000 movestogo 10",
            Color::Black,
        );
        assert_eq!(limits.time, Some(Duration::from_millis(7500)));
        assert_eq!(parse("wtime 3", Color::White).time, Some(Duration::ZERO));
        for s in ["", "infinite", "infinite nodes 1 movetime 1"] {
            let limits = parse(s, Color::White);
            assert!(limits.wait_for_stop);
            assert!(limits.nodes.is_none() && limits.time.is_none());
        }
        let limits = parse("searchmoves e2e4 d2d4 nodes 12", Color::White);
        assert_eq!(limits.searchmoves, ["e2e4", "d2d4"]);
        assert_eq!(limits.nodes, Some(12));
    }

    #[test]
    fn malformed_or_unsupported_limits_are_rejected() {
        for s in [
            "nodes",
            "nodes -1",
            "nodes 18446744073709551616",
            "movetime x",
            "movestogo 0",
            "btime 100",
            "searchmoves",
            "searchmoves nodes 5",
            "ponder",
            "depth 5",
            "mate 1",
        ] {
            assert!(
                Limits::parse(s.split_whitespace(), Color::White).is_err(),
                "{s}"
            );
        }
    }
}
