"""Inspect the transformer trunk's shapes and parameter ownership on CPU."""

import argparse

import torch

from neurodiktyon import FEATURE_COUNT, SQUARE_COUNT, TransformerTrunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    trunk = TransformerTrunk(
        n_blocks=args.n_blocks,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    features = torch.randn(2, SQUARE_COUNT, FEATURE_COUNT)
    stages = [("Input features", features)]
    with torch.inference_mode():
        tokens = trunk.embedding(features)
        stages.append(("Square embedding", tokens))
        for index, block in enumerate(trunk.blocks, start=1):
            tokens = block(tokens)
            stages.append((f"Block {index}", tokens))
        output = trunk(features)
        torch.testing.assert_close(output, trunk.final_norm(tokens))
        stages.append(("Final LayerNorm (output)", output))

    print("Synthetic features; not encoded legal positions or a trained chess model.")
    print(
        f"Blocks: {args.n_blocks}; d_model: {args.d_model}; "
        f"heads: {args.n_heads}; d_ff: {args.d_ff}; GAB: 8 / 32 / 32"
    )
    for name, tensor in stages:
        print(f"{name:27} {list(tensor.shape)}")

    shared = sum(p.numel() for p in trunk.templates.parameters())
    per_block = [
        sum(p.numel() for p in block.parameters()) - shared for block in trunk.blocks
    ]
    print(
        f"Embedding parameters: {sum(p.numel() for p in trunk.embedding.parameters()):,}"
    )
    print(f"Independent parameters per block: {per_block[0]:,}")
    print(f"Independent parameters across all blocks: {sum(per_block):,}")
    print(f"Shared template parameters (counted once): {shared:,}")
    print(
        f"Final LayerNorm parameters: {sum(p.numel() for p in trunk.final_norm.parameters()):,}"
    )
    print(f"Total trunk parameters: {sum(p.numel() for p in trunk.parameters()):,}")
    print(
        "Every block references the same template bank: "
        f"{all(block.gab.templates is trunk.templates for block in trunk.blocks)}"
    )


if __name__ == "__main__":
    main()
