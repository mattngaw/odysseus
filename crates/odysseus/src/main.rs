use std::io;

mod uci;

fn main() -> io::Result<()> {
    uci::Session::default().run(io::BufReader::new(io::stdin()), io::stdout().lock())
}
