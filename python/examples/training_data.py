"""Inspect a Rust-exported JSONL file and its CPU training tensors."""

import argparse

from neurodiktyon.training_data import collate_examples, read_games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="completed-game JSONL from self_play_export")
    args = parser.parse_args()
    for number, game in enumerate(read_games(args.path), start=1):
        print(f"Game {number}: {game.adjudication}, winner={game.winner}")
        if not game.examples:
            print("  No searched roots (terminal starting position).")
            continue
        batch = collate_examples(game.examples)
        for field, tensor in zip(batch._fields, batch, strict=True):
            print(f"  {field}: {list(tensor.shape)}, {tensor.dtype}, {tensor.device}")
        for row, example in enumerate(game.examples[:4]):
            print(
                f"  Ply {example.ply}, {example.side_to_move}: "
                f"{len(example.legal_indices)} legal, "
                f"{sum(count > 0 for count in example.visits)} visited, "
                f"{example.total_visits} raw visits, "
                f"policy sum={batch.policy_targets[row].sum().item():.6f}, "
                f"W/D/L={list(example.value_target)}"
            )


if __name__ == "__main__":
    main()
