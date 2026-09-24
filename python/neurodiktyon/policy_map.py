"""Gather pair and promotion logits into Pyxis's fixed vocabulary order."""

import torch
from torch import Tensor, nn

from ._policy_map_data import BASE_INDICES, PROMOTION_INDICES
from .embedding import SQUARE_COUNT

BASE_MOVE_COUNT = len(BASE_INDICES)
POLICY_SIZE = BASE_MOVE_COUNT + len(PROMOTION_INDICES)


class PolicyMap(nn.Module):
    """Gather [B, 64, 64] and [B, 8, 8, 3] logits into [B, 1858].

    Pair axes are source/destination squares. Promotion axes are source file,
    destination file, and queen/rook/bishop. Both use current-player coordinates.
    Knight promotions and king-to-rook castling use the base pair entries.
    Index buffers move with the module and are saved in its state_dict.
    This module has no learned parameters, legality checks, or normalization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "base_indices", torch.tensor(BASE_INDICES, dtype=torch.long)
        )
        self.register_buffer(
            "promotion_indices", torch.tensor(PROMOTION_INDICES, dtype=torch.long)
        )

    def forward(self, pair_logits: Tensor, promotion_logits: Tensor) -> Tensor:
        if pair_logits.ndim != 3 or pair_logits.shape[1:] != (
            SQUARE_COUNT,
            SQUARE_COUNT,
        ):
            raise ValueError(
                f"expected pair_logits [batch, {SQUARE_COUNT}, {SQUARE_COUNT}], "
                f"got {tuple(pair_logits.shape)}"
            )
        expected = (pair_logits.shape[0], 8, 8, 3)
        if promotion_logits.shape != expected:
            raise ValueError(
                f"expected promotion_logits shape {expected}, "
                f"got {tuple(promotion_logits.shape)}"
            )
        base = pair_logits.flatten(start_dim=1).index_select(1, self.base_indices)
        promotions = promotion_logits.flatten(start_dim=1).index_select(
            1, self.promotion_indices
        )
        return torch.cat((base, promotions), dim=1)
