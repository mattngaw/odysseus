"""Run the complete model on synthetic features and inspect both outputs on CPU."""

import argparse

import torch

from neurodiktyon import FEATURE_COUNT, SQUARE_COUNT, ChessModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--d-policy", type=int, default=256)
    parser.add_argument("--d-value-hidden", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = ChessModel(**vars(args)).eval()
    features = torch.randn(2, SQUARE_COUNT, FEATURE_COUNT)
    with torch.inference_mode():
        output = model(features)
        wdl = output.value_logits.softmax(dim=-1)
        values = wdl[:, 0] - wdl[:, 2]

    print("Synthetic features and random weights; no trained chess predictions.")
    print(
        f"Blocks: {args.n_blocks}; d_model: {args.d_model}; "
        f"heads: {args.n_heads}; d_ff: {args.d_ff}; GAB: 8 / 32 / 32"
    )
    print(f"Policy width: {args.d_policy}; value hidden width: {args.d_value_hidden}")
    print(f"Input features: {list(features.shape)}")
    print(f"output.policy_logits: {list(output.policy_logits.shape)} (Pyxis order)")
    print(f"output.value_logits:  {list(output.value_logits.shape)} (W/D/L order)")
    for name, component in [
        ("Shared trunk", model.trunk),
        ("Policy head", model.policy_head),
        ("Value head", model.value_head),
        ("Complete model", model),
    ]:
        print(
            f"{name:16} {sum(p.numel() for p in component.parameters()):>10,} parameters"
        )
    print(
        f"Batch 0, e2e4 policy logit (index 322): {output.policy_logits[0, 322]:+.6f}"
    )
    print("Value conversion below is performed by this demo, outside the model.")
    for index in range(features.shape[0]):
        print(f"Batch {index} W/D/L logits: {output.value_logits[index].tolist()}")
        win, draw, loss = wdl[index].tolist()
        print(
            f"  P(win)={win:.6f}, P(draw)={draw:.6f}, P(loss)={loss:.6f}; "
            f"p_win - p_loss={values[index].item():+.6f}"
        )
    print("Legal-move selection and policy softmax remain outside the model.")


if __name__ == "__main__":
    main()
