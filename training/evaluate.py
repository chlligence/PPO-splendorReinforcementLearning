"""Head-to-head evaluation for Splendor agents.

Evaluates two models by playing N games and computing win rates.
Used for ELO estimation and tracking training progress.
"""

from typing import Optional, Tuple
import numpy as np

from splendor.env import SplendorEnv
from splendor.card import load_cards


def evaluate_head_to_head(
    model_a,              # SB3 MaskablePPO or None (random)
    model_b,              # SB3 MaskablePPO or None (random)
    cards_path: str,
    num_games: int = 200,
    deterministic: bool = False,
) -> dict:
    """Evaluate two models against each other.

    Each iteration plays 2 games with swapped sides to eliminate first-player
    advantage. So num_matches=100 produces 200 actual game results.

    Args:
        model_a: First model (MaskablePPO instance or None for random).
        model_b: Second model (MaskablePPO instance or None for random).
        cards_path: Path to cards_data.xlsx.
        num_games: Number of side-swapped match pairs (×2 = actual games).
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
    # Track per-seat results for detecting seat asymmetry
    a_wins_as_p0 = 0  # model_a won while sitting at P0
    a_wins_as_p1 = 0  # model_a won while sitting at P1

    for match_idx in range(num_games):
        if match_idx > 0 and match_idx % 20 == 0:
            print(f"  [Eval] Game {match_idx}/{num_games}...")

        # Both games in a match pair share the SAME seed so the card shuffle
        # is identical — the only difference is who sits where.  This cancels
        # deck-draw luck rather than adding independent noise on top of it.
        #
        # P0 always starts first (matching training).  Alternating the starting
        # player would inject OOD games: "P1-seat starts first" never occurs in
        # the post-seat-fix training distribution, so it only adds noise.
        #
        # NOTE: seeds 0..N-1 are reused every generation — the same set of
        # decks is evaluated each time.  This makes inter-generation win rates
        # directly comparable with low variance, but all ELO estimates are
        # conditioned on this particular set of decks.  For independent sampling
        # across generations, use:  match_seed = generation * 100_000 + match_idx
        match_seed = match_idx

        # Game 1: model_a=P0, model_b=P1
        obs, info = env.reset(seed=match_seed)

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

            if terminated or truncated:
                winner = env.state.winner
                if winner == 0:
                    a_wins += 1
                    a_wins_as_p0 += 1  # model_a sat at P0 in Game 1
                elif winner == 1:
                    b_wins += 1
                else:
                    draws += 1
                total_turns += env.state.turn_number
                break

        # Game 2: swap sides — model_b=P0, model_a=P1.  Same seed → same
        # card deal as Game 1, only the seat assignments are swapped.
        # P0 still starts first (set by SplendorEnv default).
        obs, info = env.reset(seed=match_seed)

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

            if terminated or truncated:
                winner = env.state.winner
                if winner == 0:
                    b_wins += 1  # model_b was P0
                elif winner == 1:
                    a_wins += 1  # model_a was P1
                    a_wins_as_p1 += 1
                else:
                    draws += 1
                total_turns += env.state.turn_number
                break

    total_games = num_games * 2  # Each match = 2 games (side swap)
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_wins / total_games,
        "b_win_rate": b_wins / total_games,
        "draw_rate": draws / total_games,
        "avg_turns": total_turns / total_games,
        # Per-seat breakdown: helps detect seat asymmetry in training
        "a_wins_as_p0": a_wins_as_p0,
        "a_wins_as_p1": a_wins_as_p1,
    }


def estimate_elo(
    model,                # Model to evaluate
    pool,                 # OpponentPool instance
    cards_path: str,
    num_games: int = 200,
    elo_k: float = 32.0,
    prior_elo: Optional[float] = None,
) -> float:
    """Estimate a model's ELO by playing against pool members.

    Plays against the top 2 pool entries (by ELO) for calibration.

    Args:
        model: Model to evaluate.
        pool: OpponentPool with historical entries.
        cards_path: Path to cards_data.xlsx.
        num_games: Games per opponent.
        elo_k: ELO K-factor for updates.
        prior_elo: The agent's own last known ELO to update from. If None,
            resets to the 1200 starting baseline (only appropriate the very
            first time a model is ever evaluated).

    Returns:
        Estimated ELO rating.
    """
    if pool.size() == 0:
        return 1200.0  # Starting ELO

    # Pick top entries by ELO for calibration
    sorted_entries = sorted(pool.entries, key=lambda e: e.elo, reverse=True)
    calibration_opponents = sorted_entries[:min(2, len(sorted_entries))]

    current_elo = prior_elo if prior_elo is not None else 1200.0

    for entry in calibration_opponents:
        from sb3_contrib import MaskablePPO
        print(f"  [ELO] vs opponent gen {entry.generation} (ELO: {entry.elo:.0f})...")
        opp_model = MaskablePPO.load(entry.path, device="cpu")

        results = evaluate_head_to_head(
            model, opp_model, cards_path,
            num_games=max(10, num_games // len(calibration_opponents)),
        )

        # ELO update
        expected = 1.0 / (1.0 + 10 ** ((entry.elo - current_elo) / 400.0))
        actual = results["a_win_rate"] + 0.5 * results["draw_rate"]
        current_elo += elo_k * (actual - expected)

    return current_elo


def evaluate_generation(
    agent_model,
    pool,                 # OpponentPool instance
    cards_path: str,
    generation: int,
    last_known_elo: float,
    eval_interval_generations: int,
    cheap_eval_games: int,
    cheap_elo_k: float,
    full_eval_games: int,
    full_elo_k: float = 32.0,
) -> dict:
    """Evaluate one generation's checkpoint: a cheap real measurement vs the
    latest pool entry every call, plus a full calibrated evaluation on
    interval boundaries that overwrites the cheap estimate. Never fabricates
    a placeholder value — every result is either a real measurement or (only
    when the pool is empty, i.e. generation 0) an explicit baseline.

    Shared by the live training loop (self_play_loop.run_self_play) and the
    offline pool-index rebuild script, so both use identical eval logic.

    Returns:
        Dict with keys: elo, elo_source, win_rate_vs_prev, win_rate_source.
    """
    latest_entry = pool.get_latest_entry()
    if latest_entry is None:
        return {
            "elo": 1200.0,
            "elo_source": "baseline",
            "win_rate_vs_prev": None,
            "win_rate_source": "baseline",
        }

    from sb3_contrib import MaskablePPO
    latest_opp = MaskablePPO.load(latest_entry.path, device="cpu")

    print("Cheap eval vs latest opponent...")
    cheap_results = evaluate_head_to_head(
        agent_model, latest_opp, cards_path, num_games=cheap_eval_games,
    )
    win_rate_vs_prev = cheap_results["a_win_rate"]
    win_rate_source = "cheap_vs_latest"
    print(f"  Per-seat: as P0={cheap_results['a_wins_as_p0']}/{cheap_eval_games} "
          f"as P1={cheap_results['a_wins_as_p1']}/{cheap_eval_games}")

    expected = 1.0 / (1.0 + 10 ** ((latest_entry.elo - last_known_elo) / 400.0))
    actual = cheap_results["a_win_rate"] + 0.5 * cheap_results["draw_rate"]
    elo = last_known_elo + cheap_elo_k * (actual - expected)
    elo_source = "incremental_vs_latest"
    print(f"  Win rate vs latest: {win_rate_vs_prev:.2%}, incremental ELO: {elo:.0f}")

    if generation % eval_interval_generations == 0:
        print("Full evaluation (ELO estimation)...")
        elo = estimate_elo(
            agent_model, pool, cards_path,
            num_games=full_eval_games, elo_k=full_elo_k,
            prior_elo=last_known_elo,
        )
        elo_source = "full_eval"
        print(f"  Estimated ELO (full): {elo:.0f}")

        full_results = evaluate_head_to_head(
            agent_model, latest_opp, cards_path, num_games=50,
        )
        win_rate_vs_prev = full_results["a_win_rate"]
        win_rate_source = "full_eval"
        print(f"  Win rate vs latest opponent (full): {win_rate_vs_prev:.2%}")

    return {
        "elo": elo,
        "elo_source": elo_source,
        "win_rate_vs_prev": win_rate_vs_prev,
        "win_rate_source": win_rate_source,
    }
