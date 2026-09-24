"""Policy logits from source/destination scores and promotion adjustments."""

import math

from torch import Tensor, nn

from .embedding import SQUARE_COUNT
from .policy_map import PolicyMap


class PolicyPairScorer(nn.Module):
    """Map [batch, 64, d_model] tokens to raw [batch, from, to] logits.

    Separate biased projections produce source and destination vectors of
    width d_policy. Each pair's logit is their dot product / sqrt(d_policy).
    Square indices retain the input's current-player coordinates, a1 through h8.

    All 64 x 64 pairs are scored, including same-square and illegal pairs.
    Vocabulary gathering, promotion adjustments, and legal-move softmax belong
    outside this component. There is no activation or dropout here.
    """

    def __init__(self, d_model: int, d_policy: int = 256) -> None:
        super().__init__()
        if d_model <= 0 or d_policy <= 0:
            raise ValueError("d_model and d_policy must be positive")

        self.d_model = d_model
        self.d_policy = d_policy
        self.scale = 1.0 / math.sqrt(d_policy)
        self.from_proj = nn.Linear(d_model, d_policy)
        self.to_proj = nn.Linear(d_model, d_policy)

    def forward(self, tokens: Tensor) -> Tensor:
        logits, _ = self.forward_with_destinations(tokens)
        return logits

    def forward_with_destinations(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Return pair logits and their [batch, 64, d_policy] destination vectors.

        The promotion branch can reuse these vectors without another projection.
        Both outputs retain their gradient connections to tokens and parameters.
        """
        if tokens.ndim != 3 or tokens.shape[1:] != (SQUARE_COUNT, self.d_model):
            raise ValueError(
                f"expected [batch, {SQUARE_COUNT}, {self.d_model}], "
                f"got {tuple(tokens.shape)}"
            )

        sources = self.from_proj(tokens)
        destinations = self.to_proj(tokens)
        logits = (sources @ destinations.transpose(-2, -1)) * self.scale
        return logits, destinations


class PromotionScorer(nn.Module):
    """Add destination-conditioned Q/R/B offsets to promotion pair logits.

    Inputs are already-scaled [batch, 64, 64] pair logits and the associated
    [batch, 64, d_policy] destination vectors, in current-player coordinates.
    Output: [batch, 8, 8, 3], with axes rank-7 source file, rank-8 destination
    file, and queen/rook/bishop. Knight promotion keeps the original pair logit.

    This includes every file pair. Selecting the 22 possible promotion routes
    and gathering the vocabulary belong outside this component.
    """

    def __init__(self, d_policy: int) -> None:
        super().__init__()
        if d_policy <= 0:
            raise ValueError("d_policy must be positive")
        self.d_policy = d_policy
        # Rows produce Q, R, B, and a common adjustment relative to knight.
        self.projection = nn.Linear(d_policy, 4, bias=False)

    def offsets(self, destinations: Tensor) -> Tensor:
        """Return [batch, 8, 3] Q/R/B offsets shared across source files."""
        if destinations.ndim != 3 or destinations.shape[1:] != (
            SQUARE_COUNT,
            self.d_policy,
        ):
            raise ValueError(
                f"expected destinations [batch, {SQUARE_COUNT}, {self.d_policy}], "
                f"got {tuple(destinations.shape)}"
            )
        raw = self.projection(destinations[:, 56:64])  # Relative rank 8.
        return raw[..., :3] + raw[..., 3:4]

    def forward(self, pair_logits: Tensor, destinations: Tensor) -> Tensor:
        offsets = self.offsets(destinations)
        expected = (offsets.shape[0], SQUARE_COUNT, SQUARE_COUNT)
        if pair_logits.shape != expected:
            raise ValueError(
                f"expected pair_logits shape {expected}, got {tuple(pair_logits.shape)}"
            )
        # The pair scores already contain 1/sqrt(d_policy). Offsets are unscaled.
        base = pair_logits[:, 48:56, 56:64].unsqueeze(-1)
        return base + offsets.unsqueeze(1)


class PolicyHead(nn.Module):
    """Map [batch, 64, d_model] trunk tokens to raw [batch, 1858] logits.

    Tokens use current-player coordinates. Pair scoring and promotion scoring
    share the destination projection; PolicyMap supplies Pyxis vocabulary order.
    Legal-move selection and softmax belong outside the head.
    """

    def __init__(self, d_model: int, d_policy: int = 256) -> None:
        super().__init__()
        self.pairs = PolicyPairScorer(d_model, d_policy)
        self.promotions = PromotionScorer(d_policy)
        self.mapping = PolicyMap()

    def forward(self, tokens: Tensor) -> Tensor:
        pair_logits, destinations = self.pairs.forward_with_destinations(tokens)
        promotion_logits = self.promotions(pair_logits, destinations)
        return self.mapping(pair_logits, promotion_logits)
