"""Head-to-head evaluation for Splendor agents.

Evaluates two models by playing N games and computing win rates.
Used for ELO estimation and tracking training progress.
"""

from typing import Optional, Tuple
import numpy as np

from splendor.env import SplendorEnv
from splendor.card import load_cards
from splendor.action_mask import get_action_mask
from splendor.observation import build_observation


def evaluate_head_to_head(
    model_a,              # SB3 MaskablePPO or None (random)
    model_b,              # SB3 MaskablePPO or None (random)
    cards_path: str,
    num_games: int = 200,
    deterministic: bool = False,
) -> dict:
    """Evaluate two models against each other.

    Each model plays as both P0 and P1 for an equal number of games.

    Args:
        model_a: First model (MaskablePPO instance or None for random).
        model_b: Second model (MaskablePPO instance or None for random).
        cards_path: Path to cards_data.xlsx.
        num_games: Total number of evaluation games.
        deterministic: If True, use greedy policy (no sampling).

    Returns:
        Dict with keys:
            a_wins, b_wins, draws: win counts
            a_win_rate, b_win_rate, draw_rate: percentages
            avg_turns: average game length
    """
    cards = load_cards(cards_path)
    env = SplendorEnv(cards)
    rng = np.random.default_rng()

    a_wins = 0
    b_wins = 0
    draws = 0
    total_turns = 0

    for game in range(num_games):
        # Alternate who starts
        starting_player = game % 2
        obs, info = env.reset(seed=game)

        # Override starting player after reset
        env.state.current_player = starting_player
        obs = build_observation(env.state, starting_player)
        info["action_mask"] = get_action_mask(env.state, starting_player)

        while True:
            current = env.state.current_player
            model = model_a if current == 0 else model_b
            mask = info["action_mask"]

            if model is not None:
                action, _ = model.predict(
                    obs,
                    action_masks=mask,
                    deterministic=deterministic,
                )
            else:
                legal = np.where(mask)[0]
                if len(legal) == 0:
                    legal = np.array([50])  # fallback pass
                action = rng.choice(legal)

            obs, reward, terminated, truncated, info = env.step(int(action))

            if terminated:
                winner = env.state.winner
                if winner == 0:
                    a_wins += 1
                elif winner == 1:
                    b_wins += 1
                else:
                    draws += 1
                total_turns += env.state.turn_number
                break

        # Second half: swap sides
        obs, info = env.reset(seed=game + 100000)

        env.state.current_player = starting_player
        obs = build_observation(env.state, starting_player)
        info["action_mask"] = get_action_mask(env.state, starting_player)

        while True:
            current = env.state.current_player
            # Swap: model_a is P1, model_b is P0 in the second half
            model = model_b if current == 0 else model_a
            mask = info["action_mask"]

            if model is not None:
                action, _ = model.predict(
                    obs, action_masks=mask, deterministic=deterministic,
                )
            else:
                legal = np.where(mask)[0]
                if len(legal) == 0:
                    legal = np.array([50])
                action = rng.choice(legal)

            obs, reward, terminated, truncated, info = env.step(int(action))

            if terminated:
                winner = env.state.winner
                if winner == 0:
                    b_wins += 1  # model_b was P0
                elif winner == 1:
                    a_wins += 1  # model_a was P1
                else:
                    draws += 1
                total_turns += env.state.turn_number
                break

    total = num_games * 2  # 2 games per iteration (side swap)
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_wins / total,
        "b_win_rate": b_wins / total,
        "draw_rate": draws / total,
        "avg_turns": total_turns / total,
    }


def estimate_elo(
    model,                # Model to evaluate
    pool,                 # OpponentPool instance
    cards_path: str,
    num_games: int = 200,
    elo_k: float = 32.0,
) -> float:
    """Estimate a model's ELO by playing against pool members.

    Plays against the top 3 pool entries (by ELO) for calibration.

    Args:
        model: Model to evaluate.
        pool: OpponentPool with historical entries.
        cards_path: Path to cards_data.xlsx.
        num_games: Games per opponent.
        elo_k: ELO K-factor for updates.

    Returns:
        Estimated ELO rating.
    """
    if pool.size() == 0:
        return 1200.0  # Starting ELO

    # Pick top entries by ELO for calibration
    sorted_entries = sorted(pool.entries, key=lambda e: e.elo, reverse=True)
    calibration_opponents = sorted_entries[:min(3, len(sorted_entries))]

    current_elo = 1200.0

    for entry in calibration_opponents:
        from sb3_contrib import MaskablePPO
        opp_model = MaskablePPO.load(entry.path)

        results = evaluate_head_to_head(
            model, opp_model, cards_path,
            num_games=max(10, num_games // len(calibration_opponents)),
        )

        # ELO update
        expected = 1.0 / (1.0 + 10 ** ((entry.elo - current_elo) / 400.0))
        actual = results["a_win_rate"] + 0.5 * results["draw_rate"]
        current_elo += elo_k * (actual - expected)

    return current_elo
