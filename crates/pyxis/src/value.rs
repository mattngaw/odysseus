use std::ops::Neg;

/// A finite expected game outcome in [-1, 1], from one player's perspective.
///
/// Evaluators return values from the side-to-move's perspective. Negation switches
/// to the opponent's perspective; the value itself does not store a player.
/// A value of -1 denotes a loss and +1 a win for that player.
/// A neutral value of zero can reflect either a draw or uncertainty about the winner.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Value(f32);

impl Value {
    /// Returns `None` for NaN, infinities, or values outside [-1, 1].
    pub const fn new(value: f32) -> Option<Self> {
        if value >= -1.0 && value <= 1.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    #[test]
    fn valid_values_are_preserved_exactly() {
        for raw in [
            -1.0,
            (-1.0_f32).next_up(),
            -0.75,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            0.25,
            1.0_f32.next_down(),
            1.0,
        ] {
            assert_eq!(Value::new(raw).unwrap().get().to_bits(), raw.to_bits());
        }
    }

    #[test]
    fn invalid_values_are_rejected() {
        for raw in [
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -f32::MAX,
            f32::MAX,
            (-1.0_f32).next_down(),
            1.0_f32.next_up(),
        ] {
            assert_eq!(Value::new(raw), None, "accepted {raw}");
        }
    }

    #[test]
    fn negation_switches_perspective_and_is_reversible() {
        for (raw, opposite) in [
            (-1.0, 1.0),
            (-0.75, 0.75),
            (-0.0, 0.0),
            (0.0, -0.0),
            (0.25, -0.25),
            (1.0, -1.0),
        ] {
            let value = Value::new(raw).unwrap();
            assert_eq!((-value).get(), opposite);
            assert_eq!((-(-value)).get().to_bits(), raw.to_bits());
        }
    }
}
