#!/usr/bin/env python3
"""Entry point for Splendor RL training.

Usage:
    # Full self-play training (50 generations)
    python scripts/train.py

    # Baseline training vs random opponent
    python scripts/train.py --baseline

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/agent_gen_5.zip

    # Quick test with fewer steps
    python scripts/train.py --test
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import (
    CARDS_PATH, CHECKPOINT_DIR, LOG_DIR,
    SELFPLAY_CONFIG, PPO_CONFIG,
)
from training.self_play_loop import run_self_play, train_vs_random_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Train a Splendor RL agent with Self-Play PPO."
    )
    parser.add_argument(
        "--cards", type=str, default=CARDS_PATH,
        help="Path to cards_data.xlsx",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Train against random opponent only (no self-play pool)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a checkpoint file",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test run with reduced steps",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--envs", type=int, default=None,
        help="Number of parallel environments (default: config value)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Steps per generation (overrides config)",
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.cards):
        print(f"ERROR: Cards file not found: {args.cards}")
        print("Please ensure cards_data.xlsx is in the project root or data/ directory.")
        sys.exit(1)

    # Test mode: reduced settings
    if args.test:
        print("*** TEST MODE — Reduced training settings ***")
        PPO_CONFIG["n_steps"] = 256
        PPO_CONFIG["batch_size"] = 64
        SELFPLAY_CONFIG["n_envs"] = 4
        SELFPLAY_CONFIG["steps_per_generation"] = 50_000
        SELFPLAY_CONFIG["generations"] = 5
        SELFPLAY_CONFIG["eval_games"] = 20

    # Override settings from CLI
    PPO_CONFIG["device"] = args.device
    if args.envs is not None:
        SELFPLAY_CONFIG["n_envs"] = args.envs
    if args.steps is not None:
        SELFPLAY_CONFIG["steps_per_generation"] = args.steps

    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Log directory: {LOG_DIR}")

    if args.baseline:
        train_vs_random_baseline(cards_path=args.cards)
    else:
        run_self_play(
            cards_path=args.cards,
            resume_from=args.resume,
        )


if __name__ == "__main__":
    main()
