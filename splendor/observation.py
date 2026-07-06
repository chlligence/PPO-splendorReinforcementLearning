"""Observation vector builder for Splendor.

Produces a flat float32 vector of shape (203,) with all features in [0, 1].
The observation is always from the perspective of the player about to act.

Layout (203 dims total):
  Face-up cards: 12 cards × 11 features = 132
    Per card: bonus_onehot(5) + points_norm(1) + cost_norm(5)
  Deck counts: 3   (L1/40, L2/30, L3/20)
  Board gems: 6    (5 colours/4 + gold/5)
  Self tokens: 6   (/10)
  Self bonuses: 5  (/15)
  Self reserved: 3 × 11 = 33
  Self score: 1    (/15)
  Self card count: 1 (/15)
  Opponent tokens: 6 (/10)
  Opponent bonuses: 5 (/15)
  Opponent reserved count: 1 (/3)
  Opponent score: 1 (/15)
  Opponent card count: 1 (/15)
  Turn flag: 1
  Final round flag: 1
"""

import numpy as np

from .constants import (
    Gem, NUM_COLORS,
    MAX_TOTAL_TOKENS, MAX_BONUS,
    MAX_POINTS_NORM, MAX_COST_NORM,
    MAX_RESERVED_CARDS,
    CARDS_PER_LEVEL, FACE_UP_PER_LEVEL,
    MAX_GEMS_PER_COLOR, MAX_GOLD,
)
from .card import Card
from .game_state import GameState, PlayerState

# Observation space dimension
OBS_DIM = 203


def build_observation(state: GameState, player_idx: int) -> np.ndarray:
    """Build a flat observation vector from the perspective of player_idx.

    The player about to act is encoded as "self", the other as "opponent".
    This ensures the network always sees a consistent perspective regardless
    of which player it represents.

    Args:
        state: Current GameState.
        player_idx: Which player is about to act (0 or 1).

    Returns:
        float32 array of shape (203,), all values in [0, 1].
    """
    p_self = state.players[player_idx]
    p_opp = state.players[1 - player_idx]
    obs_parts = []

    # ---- Face-up cards (132) ----
    for level in [1, 2, 3]:
        face_up_list = state.face_up.get(level, [])
        for pos in range(FACE_UP_PER_LEVEL):
            if pos < len(face_up_list) and face_up_list[pos] is not None:
                card = face_up_list[pos]
                obs_parts.append(_encode_card(card))
            else:
                obs_parts.append(np.zeros(11, dtype=np.float32))

    # ---- Deck counts (3) ----
    obs_parts.append(np.array([
        len(state.decks.get(1, [])) / float(CARDS_PER_LEVEL[1]),
        len(state.decks.get(2, [])) / float(CARDS_PER_LEVEL[2]),
        len(state.decks.get(3, [])) / float(CARDS_PER_LEVEL[3]),
    ], dtype=np.float32))

    # ---- Board gems (6) ----
    board_gems = np.zeros(6, dtype=np.float32)
    for c in range(NUM_COLORS):
        board_gems[c] = state.gems_available[c] / float(MAX_GEMS_PER_COLOR)
    board_gems[Gem.GOLD] = state.gems_available[Gem.GOLD] / float(MAX_GOLD)
    obs_parts.append(board_gems)

    # ---- Self state (46) ----
    obs_parts.append(_encode_player_self(p_self))

    # ---- Opponent state (14) ----
    obs_parts.append(_encode_player_opponent(p_opp))

    # ---- Global flags (2) ----
    obs_parts.append(np.array([
        float(state.current_player),         # turn flag
        float(state.final_round_flag),        # final round indicator
    ], dtype=np.float32))

    result = np.concatenate(obs_parts)
    # Safety check
    assert len(result) == OBS_DIM, f"Expected {OBS_DIM}, got {len(result)}"
    return result.astype(np.float32)


def _encode_card(card: Card) -> np.ndarray:
    """Encode a single card as 11 normalised features.

    Layout: bonus_onehot(5) + points/MAX_POINTS_NORM(1) + cost/MAX_COST_NORM(5)
    """
    features = np.zeros(11, dtype=np.float32)

    # Bonus one-hot
    features[int(card.bonus)] = 1.0

    # Points (0–5)
    features[5] = card.points / MAX_POINTS_NORM

    # Cost vector (normalised)
    for c in range(NUM_COLORS):
        features[6 + c] = card.cost[c] / MAX_COST_NORM

    return features


def _encode_player_self(p: PlayerState) -> np.ndarray:
    """Encode the current player's state (46 dims).

    Layout:
      tokens(6)/10 + bonuses(5)/15
      + reserved_cards(3×11=33) + score/15 + card_count/15
    """
    parts = []

    # Tokens: normalise by max possible (10)
    tokens_norm = p.tokens[:6].astype(np.float32) / float(MAX_TOTAL_TOKENS)
    parts.append(tokens_norm)

    # Bonuses: normalise by reasonable upper bound
    bonuses_norm = p.bonuses[:5].astype(np.float32) / float(MAX_BONUS)
    parts.append(bonuses_norm)

    # Reserved cards (3 slots × 11 dims each)
    for slot in range(MAX_RESERVED_CARDS):
        if slot < len(p.reserved):
            parts.append(_encode_card(p.reserved[slot]))
        else:
            parts.append(np.zeros(11, dtype=np.float32))

    # Score and card count
    parts.append(np.array([
        p.points / MAX_POINTS_NORM,
        min(p.card_count, 15) / MAX_POINTS_NORM,
    ], dtype=np.float32))

    return np.concatenate(parts)


def _encode_player_opponent(p: PlayerState) -> np.ndarray:
    """Encode the opponent's state (14 dims).

    Layout:
      tokens(6)/10 + bonuses(5)/15
      + reserved_count/3(1) + score/15 + card_count/15(2)
    Note: reserved card CONTENTS are hidden (imperfect information).
    """
    parts = []

    # Tokens
    tokens_norm = p.tokens[:6].astype(np.float32) / float(MAX_TOTAL_TOKENS)
    parts.append(tokens_norm)

    # Bonuses
    bonuses_norm = p.bonuses[:5].astype(np.float32) / float(MAX_BONUS)
    parts.append(bonuses_norm)

    # Reserved count only (not card contents — hidden information)
    parts.append(np.array([
        len(p.reserved) / float(MAX_RESERVED_CARDS),
    ], dtype=np.float32))

    # Score and card count
    parts.append(np.array([
        p.points / MAX_POINTS_NORM,
        min(p.card_count, 15) / MAX_POINTS_NORM,
    ], dtype=np.float32))

    return np.concatenate(parts)
