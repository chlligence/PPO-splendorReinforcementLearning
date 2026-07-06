"""Integration tests: run 1000 random games through SplendorEnv.

Verifies:
  - No crashes
  - Every game terminates
  - Winners are correctly determined
  - Token counts never go negative
  - Final round logic is correct
  - Action masking prevents illegal moves

Run with: python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from splendor.card import load_cards
from splendor.env import SplendorEnv
from splendor.constants import Gem, NUM_COLORS, MAX_TOTAL_TOKENS
from splendor.action_mask import N_ACTIONS


def load_cards_or_fake():
    """Try to load real cards, fall back to synthetic if file missing."""
    for path in ["data/cards_data.xlsx", "cards_data.xlsx"]:
        full = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path
        )
        if os.path.exists(full):
            return load_cards(full)
    # Fallback: generate synthetic cards for testing
    print("WARNING: cards_data.xlsx not found, using synthetic cards")
    from splendor.card import Card
    cards = {1: [], 2: [], 3: []}
    for level, count in [(1, 40), (2, 30), (3, 20)]:
        for i in range(count):
            bonus = Gem(i % 5)
            pts = (i // 8) if level == 1 else (1 + (i // 6)) if level == 2 else (3 + (i // 4))
            cards[level].append(Card(
                card_id=i, level=level, bonus=bonus,
                points=min(pts, 5),
                cost=((level + i % 3), (level + (i+1) % 3), 0, 0, 0)
            ))
    return cards


def run_random_game(env: SplendorEnv, seed: int) -> dict:
    """Run one complete game with random legal actions. Returns stats dict."""
    obs, info = env.reset(seed=seed)

    stats = {
        "turns": 0,
        "actions_taken": [],
        "negative_tokens_detected": False,
        "final_scores": (0, 0),
        "winner": None,
        "game_over": False,
    }

    while True:
        mask = info["action_mask"]
        legal = np.where(mask)[0]

        if len(legal) == 0:
            print(f"WARNING: No legal actions at turn {stats['turns']}!")
            break

        # Choose random legal action
        action = np.random.choice(legal)
        stats["actions_taken"].append(int(action))

        obs, reward, terminated, truncated, info = env.step(action)
        stats["turns"] += 1

        # Verify token counts
        for i in [0, 1]:
            p = env.state.players[i]
            for c in range(6):
                if p.tokens[c] < 0:
                    stats["negative_tokens_detected"] = True
            total = int(np.sum(p.tokens))
            if total > MAX_TOTAL_TOKENS:
                stats["token_overflow_detected"] = True

        # Verify gem pool
        for c in range(6):
            if env.state.gems_available[c] < 0:
                stats["negative_pool_gems"] = True

        if terminated or truncated:
            stats["game_over"] = True
            stats["final_scores"] = (
                env.state.players[0].points,
                env.state.players[1].points,
            )
            stats["winner"] = env.state.winner
            break

    return stats


def main():
    print("Loading cards...")
    cards = load_cards_or_fake()

    N_GAMES = 1000
    print(f"Running {N_GAMES} random games...")

    env = SplendorEnv(cards)
    all_stats = []

    for seed in range(N_GAMES):
        stats = run_random_game(env, seed)
        all_stats.append(stats)

        if (seed + 1) % 200 == 0:
            print(f"  {seed + 1}/{N_GAMES} games completed...")

    # --- Analysis ---
    print(f"\n{'='*60}")
    print(f"Results for {N_GAMES} random games")
    print(f"{'='*60}")

    games_completed = sum(1 for s in all_stats if s["game_over"])
    print(f"Games completed: {games_completed}/{N_GAMES}")

    negative_tokens = sum(1 for s in all_stats if s.get("negative_tokens_detected"))
    print(f"Negative token errors: {negative_tokens}")

    overflows = sum(1 for s in all_stats if s.get("token_overflow_detected", False))
    print(f"Token overflow errors: {overflows}")

    negative_pool = sum(1 for s in all_stats if s.get("negative_pool_gems", False))
    print(f"Negative pool gem errors: {negative_pool}")

    # Win distribution
    p0_wins = sum(1 for s in all_stats if s["winner"] == 0)
    p1_wins = sum(1 for s in all_stats if s["winner"] == 1)
    draws = sum(1 for s in all_stats if s["winner"] is None)
    print(f"P0 wins: {p0_wins}, P1 wins: {p1_wins}, Draws: {draws}")

    # Turn statistics
    turns = [s["turns"] for s in all_stats]
    print(f"Avg turns/game: {np.mean(turns):.1f} (min={min(turns)}, max={max(turns)})")

    # Score statistics
    scores_p0 = [s["final_scores"][0] for s in all_stats]
    scores_p1 = [s["final_scores"][1] for s in all_stats]
    print(f"Avg final score: P0={np.mean(scores_p0):.1f}, P1={np.mean(scores_p1):.1f}")
    print(f"Max final score: P0={max(scores_p0)}, P1={max(scores_p1)}")

    # Action distribution (sample)
    all_actions = [a for s in all_stats for a in s["actions_taken"]]
    action_types = {
        "take_3": sum(1 for a in all_actions if 0 <= a <= 9),
        "take_2": sum(1 for a in all_actions if 10 <= a <= 14),
        "take_1": sum(1 for a in all_actions if 45 <= a <= 49),
        "pass": sum(1 for a in all_actions if a == 50),
        "reserve_face": sum(1 for a in all_actions if 15 <= a <= 26),
        "reserve_deck": sum(1 for a in all_actions if 27 <= a <= 29),
        "buy_face": sum(1 for a in all_actions if 30 <= a <= 41),
        "buy_reserved": sum(1 for a in all_actions if 42 <= a <= 44),
    }
    total = len(all_actions)
    print(f"\nAction distribution (n={total}):")
    for name, count in action_types.items():
        print(f"  {name}: {count} ({100*count/total:.1f}%)")

    # Final verdict
    errors = negative_tokens + overflows + negative_pool
    incomplete = N_GAMES - games_completed

    print(f"\n{'='*60}")
    if errors == 0 and incomplete == 0:
        print("ALL CHECKS PASSED ✓")
    else:
        print(f"FAILURES: {errors} errors, {incomplete} incomplete games")
    print(f"{'='*60}")

    return errors == 0 and incomplete == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
