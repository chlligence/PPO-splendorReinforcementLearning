"""Potential-based reward function for 2-player, no-nobles Splendor PPO training.

This is a reformulation of the original hand-crafted dense reward. All dense
shaping is now expressed as a single **potential-based shaping** term:

        F(s, a, s') = γ · Φ(s') − Φ(s)

By Ng, Harada & Russell (1999), adding F to the base (terminal) reward does NOT
change the optimal policy, as long as the SAME γ used in training is used here.
Over a full episode the shaping telescopes:

        Σ_t γ^t F_t  =  γ^T Φ(s_T) − Φ(s_0)  =  −Φ(s_0)   (since Φ(terminal)=0)

and Φ(s_0) ≈ 0 (empty board, 0 points, symmetric position), so the total shaping
over an episode is ≈ 0.  This removes the original design's biggest risk: dense
reward that accrued positively to *both* winner and loser and could be farmed
independently of actually winning.

The strategic ideas from the original design are preserved — they now live
inside the potential Φ instead of in per-action bonuses:

  Φ(s) = Strength(self) − Strength(opponent)          (relative, zero-sum flavor)

  Strength(p) = W_POINT   · α(turn) · p.points                 ← 分数 (后期放大)
              + W_ENGINE  · (1−progress) · engine_value(p)     ← 引擎 (前期重视)
              + W_OPP     · board_opportunity(p)               ← 可达高分牌机会

Because Φ is relative and evaluated over the *shared* board:

  * Buying a card         → self points/engine ↑, and the card leaves the board
                            (drops from opponent's opportunity) → 截胡 emerges.
  * Reserving a face-up   → card moves from shared board into *our* reserved set:
                            opponent's opportunity ↓ (blocking) while ours is kept
                            (self-importance) → reserve-blocking emerges.
  * Buy-from-reserve      → opportunity → points hand-off, no path-dependence.

α with growing turn number amplifies a *point lead* late (protect/extend lead)
and amplifies a *point deficit* late (catch-up urgency), while a tied score
contributes 0 regardless of α.

NON-shaping terms (step / pass / hoarding penalties) encode genuine action costs
and are intentionally kept OUTSIDE the invariance guarantee. They are small.
"""

from typing import Optional

import numpy as np

from .card import Card
from .constants import NUM_COLORS
from .game_state import GameState, PlayerState

# ===========================================================================
# Base (non-shaping) rewards
# ===========================================================================

TERMINAL_WIN   = 10.0
TERMINAL_LOSS  = -10.0
TERMINAL_DRAW  = 0.0

# Genuine action costs — small, deliberately outside the PBRS invariance proof.
STEP_PENALTY      = -0.005   # per non-terminal action (efficiency incentive)
PASS_PENALTY      = -0.05    # pass should be a last resort
HOARDING_PENALTY  = -0.30    # taking gems while already at/over the 10-token cap

# ===========================================================================
# Shaping configuration
# ===========================================================================

# *** MUST equal the discount factor used by the PPO trainer, or the
#     policy-invariance guarantee of PBRS no longer holds. ***
SHAPING_GAMMA = 0.99

# Include the terminal shaping correction  F = −Φ(prev)  so the shaping stream
# telescopes exactly to −Φ(s_0) ≈ 0.  If terminal-reward variance turns out too
# high in practice (a large lead makes the +10 look small), set this to False —
# you lose strict telescoping but keep a clean ±10 win/loss signal.
APPLY_TERMINAL_SHAPING = True

# Potential weights.  Ratios matter more than absolutes.  Chosen so |Φ| stays
# roughly within [-3, 3] << |terminal|, keeping the win/loss signal dominant.
W_POINT       = 0.06     # weight on α · point difference
W_ENGINE      = 0.15     # weight on early-game engine value
W_OPP         = 0.05     # weight on reachable high-point-card opportunity
OPP_CARD_CAP  = 2.0      # per-card cap on importance·affordability (variance control)

# Alpha parameters — quadratic growth from ALPHA_MIN to ALPHA_MAX.
# NOTE: verify TYPICAL_TURNS against real episode length. If 2p games end well
#       before turn 60, α never reaches ALPHA_MAX and the "late-game point rush"
#       is weaker than intended — lower TYPICAL_TURNS to match observed lengths.
ALPHA_MIN     = 0.3
ALPHA_MAX     = 5.0
TYPICAL_TURNS = 60


# ===========================================================================
# Low-level helpers (kept from the original design)
# ===========================================================================

def _relative_cost(card: Card, bonuses: np.ndarray) -> float:
    """Effective gem cost after permanent bonus discounts, floored at 0.5."""
    total = 0.0
    for c in range(NUM_COLORS):
        deficit = int(card.cost[c]) - int(bonuses[c])
        if deficit > 0:
            total += deficit
    return max(total, 0.5)  # floor prevents division by zero


def _compute_alpha(turn_number: int, typical_turns: int = TYPICAL_TURNS) -> float:
    """Phase-transition parameter α∈[ALPHA_MIN, ALPHA_MAX], quadratic in progress.

    Small early (engine building), large late (point rushing).
    """
    progress = min(float(turn_number) / float(typical_turns), 1.0)
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (progress ** 2)


def _progress(turn_number: int, typical_turns: int = TYPICAL_TURNS) -> float:
    """Game progress in [0, 1]."""
    return min(float(turn_number) / float(typical_turns), 1.0)


def _card_importance(card: Card, bonuses: np.ndarray, alpha: float) -> float:
    """重要性 = α · card.points / relative_cost.  0-point cards → 0."""
    if card.points <= 0:
        return 0.0
    return alpha * float(card.points) / _relative_cost(card, bonuses)


def _affordability_progress(player: PlayerState, card: Card) -> float:
    """How close a player is to affording a card, in [0, 1].

    Mirrors purchase logic: bonuses → colored tokens → gold wildcard.
    """
    total_needed = 0.0
    total_covered = 0.0
    for c in range(NUM_COLORS):
        deficit = max(0.0, float(card.cost[c]) - float(player.bonuses[c]))
        total_needed += deficit
        total_covered += min(deficit, float(player.tokens[c]))

    if total_needed <= 0.0:
        return 1.0

    total_covered += float(player.tokens[5])  # gold wildcard (Gem.GOLD = 5)
    return min(total_covered / total_needed, 1.0)


def _engine_value(bonuses: np.ndarray) -> float:
    """Cumulative diminishing value of a player's permanent bonus portfolio.

    Value of n bonuses in one color = Σ_{k=1..n} 1/k  (harmonic).  The marginal
    value of the n-th bonus is 1/n, so buying the first bonus of a color is worth
    1.0, the second 0.5, etc.  This naturally rewards diversifying colors early
    without any hard-coded diversity term (matches the original 1/(count+1) rule).
    """
    total = 0.0
    for c in range(NUM_COLORS):
        n = int(bonuses[c])
        for k in range(1, n + 1):
            total += 1.0 / k
    return total


# ===========================================================================
# Potential function Φ
# ===========================================================================

def _all_face_up(state: GameState) -> list:
    """Flat list of all face-up cards currently on the board."""
    cards = []
    for level in (1, 2, 3):
        cards.extend(state.face_up.get(level, []))
    return cards


def _board_opportunity(player: PlayerState, accessible_cards: list,
                       alpha: float) -> float:
    """Soft measure of a player's reachable scoring opportunity.

    Σ over accessible point-cards of  importance · affordability, each capped.

    * A card only contributes while it is *reachable* (face-up or in this
      player's reserved hand). Once bought it leaves this set and its value is
      re-expressed through the points/engine terms → no double counting.
    * Removing a card the opponent wanted (by buying/reserving it) lowers the
      opponent's opportunity → the blocking/截胡 signal is emergent.
    """
    total = 0.0
    for card in accessible_cards:
        imp = _card_importance(card, player.bonuses, alpha)
        if imp <= 0.0:
            continue
        contrib = imp * _affordability_progress(player, card)
        if contrib > OPP_CARD_CAP:
            contrib = OPP_CARD_CAP
        total += contrib
    return total


def _player_strength(player: PlayerState, accessible_cards: list,
                     alpha: float, progress: float) -> float:
    """Positional strength of a single player (higher = better)."""
    points_term = W_POINT * alpha * float(player.points)
    engine_term = W_ENGINE * (1.0 - progress) * _engine_value(player.bonuses)
    oppty_term  = W_OPP * _board_opportunity(player, accessible_cards, alpha)
    return points_term + engine_term + oppty_term


def _potential(state: GameState, player_idx: int) -> float:
    """Relative potential Φ(s) from the perspective of `player_idx`.

    Φ = Strength(self) − Strength(opponent), evaluated over the shared board
    plus each player's own reserved cards.
    """
    alpha = _compute_alpha(state.turn_number)
    progress = _progress(state.turn_number)

    me = state.players[player_idx]
    opp = state.players[1 - player_idx]

    face_up = _all_face_up(state)
    me_access = face_up + list(me.reserved)
    opp_access = face_up + list(opp.reserved)

    return (
        _player_strength(me, me_access, alpha, progress)
        - _player_strength(opp, opp_access, alpha, progress)
    )


# ===========================================================================
# Main reward function
# ===========================================================================

def compute_reward(
    prev_state: GameState,
    state: GameState,
    action: int,
    player_idx: int,
    gamma: float = SHAPING_GAMMA,
) -> float:
    """Compute the shaped reward for the player who just acted.

    reward = base_terminal + potential_based_shaping + small_action_penalties

    Args:
        prev_state: Game state BEFORE the action.
        state:      Game state AFTER the action + game-end check.
        action:     Discrete action index (0-50). Used ONLY for penalties.
        player_idx: The player who acted (0 or 1).
        gamma:      Discount factor for the shaping term. MUST match the PPO
                    trainer's gamma for the policy-invariance guarantee to hold.

    Returns:
        Scalar reward value.
    """
    # ------------------------------------------------------------------
    # 1. Terminal transition.  Φ(terminal) = 0 by convention, so the shaping
    #    term is  γ·0 − Φ(prev) = −Φ(prev), which closes the telescoping sum.
    # ------------------------------------------------------------------
    if state.game_over:
        if state.winner == player_idx:
            terminal = TERMINAL_WIN
        elif state.winner == (1 - player_idx):
            terminal = TERMINAL_LOSS
        else:
            terminal = TERMINAL_DRAW

        if APPLY_TERMINAL_SHAPING:
            terminal += -_potential(prev_state, player_idx)
        return float(terminal)

    # ------------------------------------------------------------------
    # 2. Potential-based dense shaping — this replaces ALL of the original
    #    per-action buy/reserve/blocking rewards.
    # ------------------------------------------------------------------
    shaping = gamma * _potential(state, player_idx) - _potential(prev_state, player_idx)
    reward = shaping

    # ------------------------------------------------------------------
    # 3. Small action-cost penalties (deliberately non-potential).
    # ------------------------------------------------------------------
    # Hoarding: taking gems while already at/over the token cap.
    if (0 <= action <= 14) or (45 <= action <= 49):
        if int(np.sum(prev_state.players[player_idx].tokens)) >= 10:
            reward += HOARDING_PENALTY

    # Pass.
    if action == 50:
        reward += PASS_PENALTY

    # Per-step efficiency cost (applied to every non-terminal action).
    reward += STEP_PENALTY

    return float(reward)