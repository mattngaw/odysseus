use std::fmt;

use crate::Square;

pub(crate) fn grid(f: &mut fmt::Formatter<'_>, cell: impl Fn(Square) -> char) -> fmt::Result {
    for rank in (0..8).rev() {
        write!(f, "{} ", rank + 1)?;
        for file in 0..8 {
            let square = Square::from_coords(file, rank).unwrap();
            write!(f, " {}", cell(square))?;
        }
        writeln!(f)?;
    }
    f.write_str("   a b c d e f g h")
}
