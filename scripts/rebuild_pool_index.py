#!/usr/bin/env python3
"""Offline reconstruction of pool_index.json from existing checkpoints.

The original checkpoints/pool_index.json was produced by buggy evaluation
code (see training/evaluate.py, training/opponent_pool.py history): most
generations never ran a real evaluation, and the pool's eviction policy
happened to prune out the few that did. Its win_rate_vs_prev/elo values are
not usable.

The 50 saved checkpoints (checkpoints/agent_gen_*.zip) are themselves
unaffected by those bugs — checkpoint saving happens before evaluation, and
none of the fixes touch model weights. This script replays the *fixed*
evaluation logic (training.evaluate.evaluate_generation, the same function
the live training loop now uses) over all 50 checkpoints in order, to:

  1. Reconstruct an honest pool_index.json — same bounded structure/eviction
     policy as production (opponent_pool_size), showing what the fixed
     mechanism produces on real data. Doubles as an integration test of the
     Bug1-3 fixes against a real, non-synthetic checkpoint history.
  2. Write elo_history.csv — an UNBOUNDED per-generation record (elo,
     win_rate_vs_prev, provenance) for all 50 generations regardless of
     pool eviction, useful for plotting "how strong did this run actually
     get" independent of what survives in the capped pool.

Caveat: this only fixes the retroactive bookkeeping. It cannot change which
opponents were actually sampled during the original training run — that
sampling was itself driven by the corrupted pool at the time. This is a
diagnostic/reporting tool, not a way to "fix" the completed run's training.

Usage:
    python scripts/rebuild_pool_index.py
    python scripts/rebuild_pool_index.py --checkpoint-dir checkpoints --out-dir checkpoints
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import CARDS_PATH, CHECKPOINT_DIR, SELFPLAY_CONFIG
from training.evaluate import evaluate_generation
from training.opponent_pool import OpponentPool, PoolEntry


def find_checkpoints(checkpoint_dir: str):
    """Return [(generation, path), ...] sorted by generation."""
    paths = glob.glob(os.path.join(checkpoint_dir, "agent_gen_*.zip"))
    entries = []
    for p in paths:
        m = re.search(r"agent_gen_(\d+)\.zip$", os.path.basename(p))
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda t: t[0])
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                         help="Directory containing agent_gen_*.zip checkpoints")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Where to write pool_index_rebuilt.json / elo_history.csv "
                              "(default: same as --checkpoint-dir)")
    parser.add_argument("--cards", type=str, default=CARDS_PATH)
    args = parser.parse_args()

    out_dir = args.out_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    checkpoints = find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"No agent_gen_*.zip checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)
    print(f"Found {len(checkpoints)} checkpoints: "
          f"gen {checkpoints[0][0]}..{checkpoints[-1][0]}")

    from sb3_contrib import MaskablePPO

    pool = OpponentPool(
        max_size=SELFPLAY_CONFIG["opponent_pool_size"],
        latest_prob=SELFPLAY_CONFIG["latest_opponent_prob"],
        random_prob=SELFPLAY_CONFIG["random_opponent_prob"],
    )
    last_known_elo = 1200.0
    history = []

    for generation, path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Generation {generation} ({path})")
        print(f"{'='*60}")

        agent_model = MaskablePPO.load(path, device="cpu")

        eval_result = evaluate_generation(
            agent_model, pool, args.cards, generation, last_known_elo,
            eval_interval_generations=SELFPLAY_CONFIG["eval_interval_generations"],
            cheap_eval_games=SELFPLAY_CONFIG["cheap_eval_games"],
            cheap_elo_k=SELFPLAY_CONFIG["cheap_elo_k"],
            full_eval_games=max(20, SELFPLAY_CONFIG["eval_games"] // 2),
        )
        last_known_elo = eval_result["elo"]

        # Unbounded record — every generation, regardless of pool eviction.
        history.append({
            "generation": generation,
            "elo": eval_result["elo"],
            "elo_source": eval_result["elo_source"],
            "win_rate_vs_prev": eval_result["win_rate_vs_prev"],
            "win_rate_source": eval_result["win_rate_source"],
        })

        # Bounded pool — mirrors what production training would have kept.
        pool.entries = [e for e in pool.entries if e.generation != generation]
        pool.add(PoolEntry(
            path=path,
            generation=generation,
            elo=eval_result["elo"],
            win_rate_vs_prev=eval_result["win_rate_vs_prev"],
            elo_source=eval_result["elo_source"],
            win_rate_source=eval_result["win_rate_source"],
        ))

        # Save incrementally after every generation so a long run can be
        # interrupted without losing progress.
        pool_index_out = os.path.join(out_dir, "pool_index_rebuilt.json")
        pool.save_index(pool_index_out)

        history_out = os.path.join(out_dir, "elo_history.csv")
        with open(history_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "generation", "elo", "elo_source",
                "win_rate_vs_prev", "win_rate_source",
            ])
            writer.writeheader()
            writer.writerows(history)

    print(f"\n{'='*60}")
    print("Rebuild complete.")
    print(f"Reconstructed pool (bounded, {pool.size()} entries): {pool_index_out}")
    print(f"Full ELO history ({len(history)} generations): {history_out}")
    print("These are NEW files — the original pool_index.json was left untouched.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
