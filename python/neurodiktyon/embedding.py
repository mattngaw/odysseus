"""Project Pyxis's square features into the model's hidden dimension."""

from torch import Tensor, nn

# Input contract from crates/pyxis/src/encoding.rs. The caller supplies already
# encoded features in current-player coordinates, ordered a1, b1, ..., h8.
SQUARE_COUNT = 64
FEATURE_COUNT = 110


class SquareEmbedding(nn.Module):
    """Apply the same learned affine map independently to every square.

    Input: [batch, 64, 110]. Output: [batch, 64, d_model].
    This projection neither mixes squares nor adds positional information.
    Identical feature vectors therefore produce identical embeddings.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.projection = nn.Linear(FEATURE_COUNT, d_model)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1:] != (SQUARE_COUNT, FEATURE_COUNT):
            raise ValueError(
                f"expected [batch, {SQUARE_COUNT}, {FEATURE_COUNT}], "
                f"got {tuple(features.shape)}"
            )
        return self.projection(features)
