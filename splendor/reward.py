"""Reward shaping for Splendor RL.

Terminal reward (+1/-1 for win/loss) dominates all shaping rewards.
Auxiliary rewards are kept conservative to avoid local optima.
"""

import numpy as np

from .constants import MAX_TOTAL_TOKENS, WINNING_POINTS
from .game_state import GameState


def compute_reward(
    prev_state: GameState,
    state: GameState,
    action: int,
    player_idx: int,
) -> float:
    """Compute the shaped reward for the player who just acted.

    Args:
        prev_state: Game state before the action.
        state: Game state after the action.
        action: The action index (0-44) that was taken.
        player_idx: The player who acted (0 or 1).

    Returns:
        Scalar reward value.
    """
    p = state.players[player_idx]
    p_prev = prev_state.players[player_idx]

    # ---- Terminal reward (dominates all shaping) ----
    if state.game_over:
        if state.winner == player_idx:
            return 1.0    # Win
        elif state.winner == (1 - player_idx):
            return -1.0   # Loss
        else:
            return 0.0    # Draw

    reward = 0.0

    # ---- Buy card: small positive proportional to points gained ----
    if action >= 30:  # Any buy action (face-up or reserved)
        points_gained = p.points - p_prev.points
        reward += 0.05 * points_gained

    # ---- First to trigger final round: bonus for reaching winning position ----
    if state.final_round_flag and not prev_state.final_round_flag:
        if state.final_round_player == player_idx:
            reward += 0.3

    # ---- Token efficiency penalty: discourage hoarding ----
    total_tokens = int(np.sum(p.tokens))
    if total_tokens > 8:
        reward -= 0.01 * (total_tokens - 8)

    # ---- Exact bonus match: reward for buying a card entirely with bonuses ----
    if action >= 30:
        card = _get_purchased_card(p, p_prev)
        if card is not None:
            # Check if bonuses alone cover the full cost
            exact = all(
                card.cost[c] <= int(p_prev.bonuses[c])
                for c in range(5)
            )
            if exact:
                reward += 0.02

    # ---- Penalty for buying while at max tokens ----
    if total_tokens >= MAX_TOTAL_TOKENS and action >= 30:
        reward -= 0.03

    return reward


def _get_purchased_card(p_curr, p_prev):
    """Get the card that was just purchased (if any).

    Compares the purchased lists to find the newly added card.
    """
    if len(p_curr.purchased) > len(p_prev.purchased):
        return p_curr.purchased[-1]
    return None
