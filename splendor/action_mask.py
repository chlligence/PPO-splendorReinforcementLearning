"""Action mask computation for Splendor.

Returns a boolean array of shape (45,) where True = legal action.
SB3-contrib's MaskablePPO reads the "action_mask" key from the info dict
returned by env.step() and env.reset().
"""

import numpy as np

from .constants import (
    Gem, NUM_COLORS, COMBO_3_DIFFERENT,
    MAX_RESERVED_CARDS, FACE_UP_PER_LEVEL,
)
from .game_state import GameState
from .rules import can_afford

# Total number of discrete actions
# 0-9: take 3 diff | 10-14: take 2 same | 15-26: reserve face-up | 27-29: reserve deck
# 30-41: buy face-up | 42-44: buy reserved | 45-49: take 1 gem | 50: pass
N_ACTIONS = 51


def get_action_mask(state: GameState, player_idx: int) -> np.ndarray:
    """Compute the legal-action mask for the current state and player.

    Args:
        state: Current GameState.
        player_idx: Index of the player about to act (0 or 1).

    Returns:
        Boolean array of shape (N_ACTIONS,). True entries are legal actions.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    p = state.players[player_idx]
    gems = state.gems_available

    # ---- Take 3 different gems (actions 0–9) ----
    for i, (a, b, c) in enumerate(COMBO_3_DIFFERENT):
        mask[i] = (gems[a] >= 1 and gems[b] >= 1 and gems[c] >= 1)

    # ---- Take 2 same gems (actions 10–14) ----
    # Valid only when >= 4 of that colour are available
    for color in range(NUM_COLORS):
        mask[10 + color] = (gems[color] >= 4)

    # ---- Take 1 gem (actions 45–49) ----
    # Always valid when at least 1 of that colour is available
    for color in range(NUM_COLORS):
        mask[45 + color] = (gems[color] >= 1)

    # ---- Reserve (actions 15–29) ----
    # Valid only when player has < 3 reserved cards
    can_reserve = len(p.reserved) < MAX_RESERVED_CARDS

    if can_reserve:
        # Reserve face-up card (actions 15–26): card must exist at position
        for level in [1, 2, 3]:
            face_up_list = state.face_up.get(level, [])
            for pos in range(FACE_UP_PER_LEVEL):
                action = 15 + (level - 1) * FACE_UP_PER_LEVEL + pos
                if pos < len(face_up_list) and face_up_list[pos] is not None:
                    mask[action] = True

        # Reserve from deck (actions 27–29): deck must be non-empty
        for level_idx in range(3):
            level = level_idx + 1
            action = 27 + level_idx
            if len(state.decks.get(level, [])) > 0:
                mask[action] = True

    # ---- Buy face-up card (actions 30–41) ----
    for level in [1, 2, 3]:
        face_up_list = state.face_up.get(level, [])
        for pos in range(FACE_UP_PER_LEVEL):
            action = 30 + (level - 1) * FACE_UP_PER_LEVEL + pos
            if pos < len(face_up_list) and face_up_list[pos] is not None:
                card = face_up_list[pos]
                if can_afford(p.tokens, p.bonuses, card.cost):
                    mask[action] = True

    # ---- Buy reserved card (actions 42–44) ----
    for slot in range(MAX_RESERVED_CARDS):
        action = 42 + slot
        if slot < len(p.reserved):
            card = p.reserved[slot]
            if can_afford(p.tokens, p.bonuses, card.cost):
                mask[action] = True

    # ---- Pass (action 50) ----
    # Always legal as a fallback to prevent deadlocks
    mask[50] = True

    return mask


def get_action_description(action: int) -> str:
    """Return a human-readable description of an action index.

    Useful for debugging and rendering.
    """
    if 0 <= action <= 9:
        colors = COMBO_3_DIFFERENT[action]
        names = [["Black", "White", "Red", "Blue", "Green"][c] for c in colors]
        return f"Take 3 different: {', '.join(names)}"
    elif 10 <= action <= 14:
        color = ["Black", "White", "Red", "Blue", "Green"][action - 10]
        return f"Take 2 same: {color}"
    elif 15 <= action <= 26:
        pos = action - 15
        level = (pos // 4) + 1
        slot = pos % 4
        return f"Reserve face-up L{level} pos{slot}"
    elif 27 <= action <= 29:
        level = action - 27 + 1
        return f"Reserve from deck L{level}"
    elif 30 <= action <= 41:
        pos = action - 30
        level = (pos // 4) + 1
        slot = pos % 4
        return f"Buy face-up L{level} pos{slot}"
    elif 42 <= action <= 44:
        slot = action - 42
        return f"Buy reserved slot{slot}"
    elif 45 <= action <= 49:
        color = ["Black", "White", "Red", "Blue", "Green"][action - 45]
        return f"Take 1 gem: {color}"
    elif action == 50:
        return "Pass"
    return f"Unknown action {action}"
