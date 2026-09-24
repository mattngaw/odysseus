use super::Game;
use crate::{Color, TerminalStatus};

/// An outcome recognized by [`Game::outcome`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameOutcome {
    Checkmate { winner: Color },
    Draw { reason: DrawReason },
}

/// Why the current game is recognized as drawn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrawReason {
    Stalemate,
    InsufficientMaterial,
    /// The current repetition state has occurred at least five times.
    FivefoldRepetition,
    /// At least 150 halfmoves without a pawn move or capture.
    SeventyFiveMoveRule,
}

impl Game {
    /// Reports checkmate or a recognized automatic draw.
    ///
    /// Checks, in order: checkmate/stalemate, insufficient material, fivefold
    /// repetition, and the seventy-five-move rule. Checkmate takes precedence
    /// over a draw on the 150th halfmove. If multiple draw conditions hold,
    /// returns the first reason in this order.
    ///
    /// Fivefold repetition uses [`Self::repetition_count`], including only the
    /// recorded line; the supplied starting position counts once. The move
    /// rule uses the current position's halfmove clock, including its FEN value.
    /// Material recognition uses [`crate::Position::has_insufficient_material`],
    /// with its limited scope. Claimable draws are not handled here: threefold
    /// repetition and 100 halfmoves do not automatically produce an outcome.
    /// `None` means none of the supported outcomes has been established.
    ///
    /// Recomputes from the current position and recorded line, so play and undo
    /// are reflected immediately. Does not store a result or change the position
    /// or history. Generates a complete temporary legal-move list, which may allocate.
    ///
    /// This is an observation, not a restriction on play. Neither [`Self::play`]
    /// nor [`Self::play_unchecked`] calls this query or rejects a legal move
    /// because a draw was reported. Callers are responsible for checking the
    /// outcome when deciding whether to continue. If play continues, an earlier
    /// outcome is not retained when its conditions no longer hold.
    pub fn outcome(&self) -> Option<GameOutcome> {
        match self.position.terminal_status() {
            Some(TerminalStatus::Checkmate { winner }) => Some(GameOutcome::Checkmate { winner }),
            Some(TerminalStatus::Stalemate) => Some(GameOutcome::Draw {
                reason: DrawReason::Stalemate,
            }),
            None if self.position.has_insufficient_material() => Some(GameOutcome::Draw {
                reason: DrawReason::InsufficientMaterial,
            }),
            None if self.repetition_count() >= 5 => Some(GameOutcome::Draw {
                reason: DrawReason::FivefoldRepetition,
            }),
            None if self.position.halfmove_clock() >= 150 => Some(GameOutcome::Draw {
                reason: DrawReason::SeventyFiveMoveRule,
            }),
            None => None,
        }
    }
}
