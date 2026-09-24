use penteconter::{Game, GameOutcome};

/// Why search or self-play stops at the current game state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Adjudication {
    /// An automatic outcome recognized by the chess core.
    Automatic(GameOutcome),
    /// Our baseline ends on the third occurrence, without a draw-claim action.
    /// This is a search/self-play convention, not an automatic chess-rule draw.
    ThreefoldRepetitionDraw,
}

/// Applies the shared baseline termination policy for search and self-play.
///
/// Automatic outcomes take precedence. Otherwise, the third or later occurrence
/// of the current position ends the game as a draw under this policy. Counts
/// include the supplied starting position and all recorded play/search moves.
/// No claim action or repetition penalty is modeled. This does not alter
/// [`Game::outcome`], prevent further play, or store a result.
///
/// The move-clock policy remains the core's 150-halfmove automatic draw;
/// reaching 100 halfmoves alone does not end search or self-play.
pub fn adjudicate(game: &Game) -> Option<Adjudication> {
    if let Some(outcome) = game.outcome() {
        Some(Adjudication::Automatic(outcome))
    } else if game.repetition_count() >= 3 {
        Some(Adjudication::ThreefoldRepetitionDraw)
    } else {
        None
    }
}
