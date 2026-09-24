"""Inspect one transformer block's operations, shapes, and residuals on CPU."""

import argparse
import copy

import torch

from neurodiktyon import SQUARE_COUNT, GABTemplates, TransformerBlock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    templates = GABTemplates(d3=32)
    block = TransformerBlock(args.d_model, args.n_heads, args.d_ff, templates)
    tokens = torch.randn(2, SQUARE_COUNT, args.d_model)

    with torch.inference_mode():
        normalized = block.norm_attention(tokens)
        bias = block.gab(normalized)
        attention_update = block.attention(normalized, attention_bias=bias)
        after_attention = tokens + attention_update
        normalized_ffn = block.norm_ffn(after_attention)
        ffn_update = block.ffn(normalized_ffn)
        output = block(tokens)
        torch.testing.assert_close(output, after_attention + ffn_update)

        # Disable branch outputs on a copy to expose the direct identity path.
        identity_block = copy.deepcopy(block)
        for projection in (
            identity_block.attention.out_proj,
            identity_block.ffn.out_proj,
        ):
            projection.weight.zero_()
            projection.bias.zero_()
        identity_output = identity_block(tokens)

    print("Synthetic tokens and randomly initialized weights; no trained chess model.")
    for name, tensor in [
        ("Input residual stream", tokens),
        ("LayerNorm 1 -> attention and GAB", normalized),
        ("GAB attention biases", bias),
        ("Attention update", attention_update),
        ("First residual sum", after_attention),
        ("LayerNorm 2 -> FFN", normalized_ffn),
        ("FFN update", ffn_update),
        ("Second residual sum (output)", output),
    ]:
        print(f"{name:35} {list(tensor.shape)}")
    shared = sum(p.numel() for p in templates.parameters())
    total = sum(p.numel() for p in block.parameters())
    print(f"Parameters owned by this block: {total - shared:,}")
    print(f"Shared template parameters: {shared:,}")
    print(f"Total including the template bank: {total:,}")
    print(
        f"Zero branch outputs preserve input exactly: {torch.equal(identity_output, tokens)}"
    )


if __name__ == "__main__":
    main()
