use penteconter::{PieceKind, Square};

use super::{BASE_MOVE_COUNT, POLICY_SIZE, PolicyEntry};

pub(super) struct Tables {
    pub entries: [PolicyEntry; POLICY_SIZE],
    pub base: [[u16; 64]; 64],
    pub promotions: [[[u16; 3]; 8]; 8],
}

// A small, deterministic compile-time enumeration; no startup initialization.
pub(super) static TABLES: Tables = build();

const fn build() -> Tables {
    let placeholder = PolicyEntry {
        from: Square::new(0).unwrap(),
        to: Square::new(0).unwrap(),
        promotion: None,
    };
    let mut tables = Tables {
        entries: [placeholder; POLICY_SIZE],
        base: [[u16::MAX; 64]; 64],
        promotions: [[[u16::MAX; 3]; 8]; 8],
    };
    let mut next = 0;
    let mut from = 0;
    while from < 64 {
        let mut to = 0;
        while to < 64 {
            let a = Square::new(from as u8).unwrap();
            let b = Square::new(to as u8).unwrap();
            let df = a.file().abs_diff(b.file());
            let dr = a.rank().abs_diff(b.rank());
            let ray = df == 0 || dr == 0 || df == dr;
            let knight = (df == 1 && dr == 2) || (df == 2 && dr == 1);
            if from != to && (ray || knight) {
                tables.entries[next] = PolicyEntry {
                    from: a,
                    to: b,
                    promotion: None,
                };
                tables.base[from][to] = next as u16;
                next += 1;
            }
            to += 1;
        }
        from += 1;
    }
    assert!(next == BASE_MOVE_COUNT);
    let pieces = [PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop];
    let mut from_file = 0;
    while from_file < 8 {
        let mut to_file = 0;
        while to_file < 8 {
            let from = Square::from_coords(from_file as u8, 6).unwrap();
            let to = Square::from_coords(to_file as u8, 7).unwrap();
            if from.file().abs_diff(to.file()) <= 1 {
                let mut promotion = 0;
                while promotion < 3 {
                    tables.entries[next] = PolicyEntry {
                        from,
                        to,
                        promotion: Some(pieces[promotion]),
                    };
                    tables.promotions[from_file][to_file][promotion] = next as u16;
                    next += 1;
                    promotion += 1;
                }
            }
            to_file += 1;
        }
        from_file += 1;
    }
    assert!(next == POLICY_SIZE);
    tables
}
