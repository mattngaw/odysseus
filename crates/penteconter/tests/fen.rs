use penteconter::{
    CastlingRights, Color, FenError, Piece, PieceKind, Position, PositionError, Square,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[test]
fn starting_position_has_the_expected_absolute_placement_and_metadata() {
    let position: Position = START.parse().unwrap();
    let back_rank = [
        PieceKind::Rook,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Queen,
        PieceKind::King,
        PieceKind::Bishop,
        PieceKind::Knight,
        PieceKind::Rook,
    ];
    for rank in 0..8 {
        for (file, &kind) in back_rank.iter().enumerate() {
            let expected = match rank {
                0 => Some(Piece::new(Color::White, kind)),
                1 => Some(Piece::new(Color::White, PieceKind::Pawn)),
                6 => Some(Piece::new(Color::Black, PieceKind::Pawn)),
                7 => Some(Piece::new(Color::Black, kind)),
                _ => None,
            };
            let square = Square::from_coords(file as u8, rank).unwrap();
            assert_eq!(position.board().piece_at(square), expected);
        }
    }
    assert!(position.board().is_consistent());
    assert_eq!(position.side_to_move(), Color::White);
    assert_eq!(position.castling_rights(), CastlingRights::ALL);
    assert_eq!(position.en_passant_target(), None);
    assert_eq!(position.halfmove_clock(), 0);
    assert_eq!(position.fullmove_number(), 1);
}

#[test]
fn canonical_fixtures_round_trip_including_uncapturable_en_passant() {
    // The first four positions are the PGN specification's FEN examples.
    for fen in [
        START,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 5 39",
        "7k/8/2b5/8/4P3/8/8/N3K3 w - - 0 12",
        "4k3/8/8/8/8/8/8/4K3 b - - 4294967295 4294967295",
    ] {
        let position: Position = fen.parse().unwrap();
        assert_eq!(position.to_fen(), fen);
        assert_eq!(position.to_fen().parse::<Position>().unwrap(), position);
    }
}

#[test]
fn all_castling_subsets_serialize_in_canonical_order() {
    for subset in 0u8..16 {
        let mut rights = String::new();
        for (index, symbol) in ['K', 'Q', 'k', 'q'].into_iter().enumerate() {
            if subset & (1 << index) != 0 {
                rights.push(symbol);
            }
        }
        if rights.is_empty() {
            rights.push('-');
        }
        let fen = START.replace("KQkq", &rights);
        let input = START.replace("KQkq", &rights.chars().rev().collect::<String>());
        assert_eq!(input.parse::<Position>().unwrap().to_fen(), fen);
    }
}

#[test]
fn whitespace_and_decimal_formatting_are_normalized() {
    let input = " \t4k3/8/8/8/8/8/8/4K3\n w  -  - 0005 0039\r\n";
    assert_eq!(
        input.parse::<Position>().unwrap().to_fen(),
        "4k3/8/8/8/8/8/8/4K3 w - - 5 39"
    );
}

#[test]
fn malformed_fields_return_typed_errors() {
    for input in [
        "",
        "8",
        "8/8/8/8/8/8/8/8 w - - 0",
        &format!("{START} extra"),
    ] {
        assert_eq!(input.parse::<Position>(), Err(FenError::FieldCount));
    }
    for (index, replacement, error) in [
        (0, "8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "8/8/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "8/8/8/8/8/8/8/", FenError::PiecePlacement),
        (0, "9/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "08/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "7/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "8p/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "888888888888/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "7x/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (0, "7♔/8/8/8/8/8/8/8", FenError::PiecePlacement),
        (1, "W", FenError::SideToMove),
        (1, "white", FenError::SideToMove),
        (2, "KK", FenError::CastlingRights),
        (2, "-K", FenError::CastlingRights),
        (2, "A", FenError::CastlingRights),
        (3, "E3", FenError::EnPassantTarget),
        (3, "e9", FenError::EnPassantTarget),
        (3, "e33", FenError::EnPassantTarget),
        (3, "é3", FenError::EnPassantTarget),
        (4, "-1", FenError::HalfmoveClock),
        (4, "+1", FenError::HalfmoveClock),
        (4, "4294967296", FenError::HalfmoveClock),
        (4, "１", FenError::HalfmoveClock),
        (5, "-1", FenError::FullmoveNumber),
        (5, "+1", FenError::FullmoveNumber),
        (5, "4294967296", FenError::FullmoveNumber),
    ] {
        let mut fields: Vec<_> = START.split(' ').collect();
        fields[index] = replacement;
        let fen = fields.join(" ");
        assert_eq!(fen.parse::<Position>(), Err(error), "{fen}");
    }
}

#[test]
fn structurally_invalid_positions_are_rejected_after_parsing() {
    for fen in [
        "8/8/8/8/8/8/8/8 w - - 0 1",          // Missing kings.
        "4k3/8/8/8/8/8/8/P3K3 w - - 0 1",     // Pawn on a back rank.
        "4k3/8/8/8/8/8/8/4K3 w - - 0 0",      // Fullmove zero.
        "4k3/8/8/8/8/8/8/4K3 w K - 0 1",      // Missing rook.
        "4k3/8/8/8/8/8/8/3K3R w K - 0 1",     // Displaced king.
        "4k3/8/8/8/8/8/8/4K2r w K - 0 1",     // Wrong-colored rook.
        "4k3/8/8/8/8/8/8/4K3 b - e3 0 1",     // Missing double-pushed pawn.
        "4k3/8/8/8/4p3/8/8/4K3 b - e3 0 1",   // Wrong-colored pawn.
        "4k3/8/8/8/4P3/4N3/8/4K3 b - e3 0 1", // Occupied target.
        "4k3/8/8/8/4P3/8/4N3/4K3 b - e3 0 1", // Occupied origin.
        "4k3/8/8/8/4P3/8/8/4K3 b - e3 1 1",   // Nonzero clock.
        "4k3/8/8/8/4P3/8/8/4K3 w - e3 0 1",   // Wrong side to move.
    ] {
        assert!(
            matches!(fen.parse::<Position>(), Err(FenError::InvalidPosition(_))),
            "{fen}"
        );
    }
    assert_eq!(
        "4k3/8/8/8/8/8/8/4K3 w - - 0 0".parse::<Position>(),
        Err(FenError::InvalidPosition(PositionError::ZeroFullmoveNumber))
    );
}
