use super::Position;
use crate::{CastlingSide, Color, Piece, Square};

#[cfg(test)]
mod tests;

struct FeatureKeys {
    piece_square: [[[u64; 64]; 6]; 2],
    black_to_move: u64,
    castling: [[u64; 2]; 2],
    en_passant_file: [u64; 8],
}

// 781 keys (6,248 bytes), generated once at compile time.
static KEYS: FeatureKeys = build_keys();

pub(super) fn piece_square_key(piece: Piece, square: Square) -> u64 {
    KEYS.piece_square[piece.color().index()][piece.kind().index()][square.index()]
}

impl Position {
    /// Returns the cached repetition fingerprint in constant time.
    ///
    /// Construction computes it from scratch; copy-and-apply maintains it.
    /// It has the same identity and collision caveats as
    /// [`Self::recompute_zobrist_key`], without scanning the position.
    pub const fn zobrist_key(&self) -> u64 {
        self.zobrist_key
    }

    /// Recomputes a 64-bit Zobrist fingerprint of this position's repetition identity.
    ///
    /// Includes colored piece placement, side to move, retained castling rights,
    /// and the en passant file only when at least one legal capture exists.
    /// Excludes both move counters and leaves the stored FEN target unchanged.
    /// Equality of keys is not proof of identity: collisions remain possible.
    /// This does not change `Position`'s field-by-field equality.
    ///
    /// Scans the occupied squares and, when needed, checks up to two en passant
    /// candidates. Ignores the cached key, providing a reference for checking
    /// incremental updates.
    pub fn recompute_zobrist_key(&self) -> u64 {
        let mut key = 0;
        let mut occupied = self.board.occupied();
        while let Some(square) = occupied.pop_first() {
            let piece = self
                .board
                .piece_at(square)
                .expect("occupied square has a piece");
            key ^= piece_square_key(piece, square);
        }

        key ^ self.metadata_zobrist_key()
    }

    // A small fixed set of features. EP legality is recomputed for each old/new
    // target; caching the normalized file is a separate possible optimization.
    pub(super) fn metadata_zobrist_key(&self) -> u64 {
        let mut key = 0;
        if self.side_to_move == Color::Black {
            key ^= KEYS.black_to_move;
        }
        for color in [Color::White, Color::Black] {
            for side in [CastlingSide::Kingside, CastlingSide::Queenside] {
                if self.castling_rights.contains(color, side) {
                    key ^= KEYS.castling[color.index()][side as usize];
                }
            }
        }
        if let Some(file) = self.legal_en_passant_file() {
            key ^= KEYS.en_passant_file[file as usize];
        }
        key
    }
}

const fn build_keys() -> FeatureKeys {
    // ASCII "ODYSSEUS". Key order is color, kind, square; Black to move;
    // castling by color then side; en passant files a through h.
    let mut state = 0x4f44_5953_5345_5553;
    let mut keys = FeatureKeys {
        piece_square: [[[0; 64]; 6]; 2],
        black_to_move: 0,
        castling: [[0; 2]; 2],
        en_passant_file: [0; 8],
    };
    let mut color = 0;
    while color < 2 {
        let mut kind = 0;
        while kind < 6 {
            let mut square = 0;
            while square < 64 {
                keys.piece_square[color][kind][square] = next_random(&mut state);
                square += 1;
            }
            kind += 1;
        }
        color += 1;
    }
    keys.black_to_move = next_random(&mut state);
    color = 0;
    while color < 2 {
        let mut side = 0;
        while side < 2 {
            keys.castling[color][side] = next_random(&mut state);
            side += 1;
        }
        color += 1;
    }
    let mut file = 0;
    while file < 8 {
        keys.en_passant_file[file] = next_random(&mut state);
        file += 1;
    }
    keys
}

// SplitMix64: https://prng.di.unimi.it/splitmix64.c (public domain).
// Wrapping arithmetic makes generation independent of build mode.
const fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
