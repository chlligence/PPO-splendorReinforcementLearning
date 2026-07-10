"""Pure-function game rules for Splendor.

All functions take and modify GameState in-place. No side effects beyond
the state object. This keeps the logic testable and the env thin.
"""

from typing import Optional
import numpy as np

from .constants import (
    Gem, NUM_COLORS, COMBO_3_DIFFERENT,
    MAX_RESERVED_CARDS, MAX_TOTAL_TOKENS,
    FACE_UP_PER_LEVEL, WINNING_POINTS,
)
from .card import Card
from .game_state import GameState, PlayerState


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

def execute_action(state: GameState, player_idx: int, action: int) -> None:
    """Execute a single action, mutating state in-place.

    Action encoding:
        0–9:   Take 3 different gems
        10–14: Take 2 same gems
        15–26: Reserve face-up card
        27–29: Reserve from deck
        30–41: Buy face-up card
        42–44: Buy reserved card
        45–49: Take 1 gem
        50:    Pass (no-op)

    Args:
        state: Current game state (mutated).
        player_idx: 0 or 1.
        action: Discrete action index (0–50).
    """
    if 0 <= action <= 9:
        _take_3_different(state, player_idx, action)
    elif 10 <= action <= 14:
        _take_2_same(state, player_idx, action - 10)
    elif 15 <= action <= 26:
        _reserve_face_up(state, player_idx, action - 15)
    elif 27 <= action <= 29:
        _reserve_from_deck(state, player_idx, action - 27)
    elif 30 <= action <= 41:
        _buy_face_up(state, player_idx, action - 30)
    elif 42 <= action <= 44:
        _buy_reserved(state, player_idx, action - 42)
    elif 45 <= action <= 49:
        _take_1_gem(state, player_idx, action - 45)
    elif action == 50:
        pass  # Pass — do nothing
    else:
        raise ValueError(f"Invalid action: {action}")


# ---------------------------------------------------------------------------
# Take gems
# ---------------------------------------------------------------------------

def _take_3_different(state: GameState, player_idx: int, combo_idx: int) -> None:
    """Take 1 gem each of 3 different colors."""
    colors = COMBO_3_DIFFERENT[combo_idx]
    p = state.players[player_idx]
    for c in colors:
        state.gems_available[c] -= 1
        p.tokens[c] += 1
    _auto_discard_if_needed(state, player_idx)


def _take_2_same(state: GameState, player_idx: int, color: int) -> None:
    """Take 2 gems of the same color. Only valid when >=4 available."""
    p = state.players[player_idx]
    state.gems_available[color] -= 2
    p.tokens[color] += 2
    _auto_discard_if_needed(state, player_idx)


def _take_1_gem(state: GameState, player_idx: int, color: int) -> None:
    """Take 1 gem of a single color."""
    p = state.players[player_idx]
    state.gems_available[color] -= 1
    p.tokens[color] += 1
    _auto_discard_if_needed(state, player_idx)


# ---------------------------------------------------------------------------
# Reserve
# ---------------------------------------------------------------------------

def _reserve_face_up(state: GameState, player_idx: int, position: int) -> None:
    """Reserve a face-up card at the given flat position (0-11)."""
    level = (position // FACE_UP_PER_LEVEL) + 1
    pos = position % FACE_UP_PER_LEVEL
    card = state.face_up[level][pos]
    state.face_up[level][pos] = None
    _add_reserved_card(state, player_idx, card)
    # Draw replacement immediately for reserves
    if state.decks.get(level) and len(state.decks[level]) > 0:
        state.face_up[level][pos] = state.decks[level].pop()


def _reserve_from_deck(state: GameState, player_idx: int, level_idx: int) -> None:
    """Reserve the top card from a deck (blind draw)."""
    level = level_idx + 1  # 0->1, 1->2, 2->3
    if state.decks[level]:
        card = state.decks[level].pop()  # top of deck
    else:
        card = None  # Should never happen if action mask is correct
    if card is not None:
        _add_reserved_card(state, player_idx, card)


def _add_reserved_card(state: GameState, player_idx: int, card: Card) -> None:
    """Add a card to player's reserved hand, grant gold if available."""
    p = state.players[player_idx]
    p.reserved.append(card)
    # Grant 1 gold if available
    if state.gems_available[Gem.GOLD] > 0:
        state.gems_available[Gem.GOLD] -= 1
        p.tokens[Gem.GOLD] += 1
    _auto_discard_if_needed(state, player_idx)


# ---------------------------------------------------------------------------
# Buy
# ---------------------------------------------------------------------------

def _buy_face_up(state: GameState, player_idx: int, position: int) -> None:
    """Buy a face-up card at the given flat position (0-11)."""
    level = (position // FACE_UP_PER_LEVEL) + 1
    pos = position % FACE_UP_PER_LEVEL
    card = state.face_up[level][pos]
    state.face_up[level][pos] = None
    _purchase_card(state, player_idx, card)
    # Draw replacement from deck for face-up slot
    if state.decks.get(level) and len(state.decks[level]) > 0:
        state.face_up[level][pos] = state.decks[level].pop()


def _buy_reserved(state: GameState, player_idx: int, slot: int) -> None:
    """Buy a card from the player's reserved hand."""
    p = state.players[player_idx]
    card = p.reserved.pop(slot)
    _purchase_card(state, player_idx, card)
    # No replacement drawn — reserved cards are from hand


def _purchase_card(state: GameState, player_idx: int, card: Card) -> None:
    """Execute the card purchase: pay tokens, claim card, update state.

    Does NOT draw a replacement — callers handle board slot management.
    """
    p = state.players[player_idx]

    # 1. Compute effective cost after bonus discounts
    effective_cost = np.zeros(NUM_COLORS, dtype=np.int32)
    for c in range(NUM_COLORS):
        effective_cost[c] = max(0, card.cost[c] - int(p.bonuses[c]))

    # 2. Pay with coloured tokens first
    for c in range(NUM_COLORS):
        pay = min(effective_cost[c], int(p.tokens[c]))
        p.tokens[c] -= pay
        state.gems_available[c] += pay
        effective_cost[c] -= pay

    # 3. Cover remaining deficit with gold (wildcard)
    gold_needed = int(np.sum(effective_cost))
    p.tokens[Gem.GOLD] -= gold_needed
    state.gems_available[Gem.GOLD] += gold_needed

    # 4. Claim the card
    p.purchased.append(card)
    p.bonuses[card.bonus] += 1
    p.points += card.points
    p.card_count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Affordability check
# ---------------------------------------------------------------------------

def can_afford(tokens: np.ndarray, bonuses: np.ndarray,
               cost: tuple) -> bool:
    """Check if a player can afford a card given tokens, bonuses, and gold.

    Gold (index 5) acts as a wildcard: can substitute 1:1 for any missing colour.

    Args:
        tokens: Player's token counts, shape (6,).
        bonuses: Player's permanent bonuses, shape (5,).
        cost: Card cost as (black, white, red, blue, green) tuple.

    Returns:
        True if the player has enough resources to purchase the card.
    """
    gold_available = int(tokens[Gem.GOLD])
    total_deficit = 0
    for c in range(NUM_COLORS):
        deficit = max(0, int(cost[c]) - int(bonuses[c]) - int(tokens[c]))
        total_deficit += deficit
    return total_deficit <= gold_available


# ---------------------------------------------------------------------------
# Token discard (auto-handle overflow)
# ---------------------------------------------------------------------------

def _auto_discard_if_needed(state: GameState, player_idx: int) -> None:
    """Automatically discard tokens if player exceeds MAX_TOTAL_TOKENS.

    Discard priority (least valuable → most valuable):
        1. Colours with highest count (excess of one colour is least useful)
        2. Tiebreak among colours: BLACK → WHITE → RED → BLUE → GREEN
        3. Gold LAST (万能, most valuable — substitutes any colour)
    This is deterministic so the agent can learn to manage tokens.
    """
    p = state.players[player_idx]
    total = int(np.sum(p.tokens))
    excess = total - MAX_TOTAL_TOKENS

    while excess > 0:
        # Priority 1-2: discard colour with most tokens, tiebreak by index
        max_count = 0
        discard_color = -1
        for c in range(NUM_COLORS):
            if p.tokens[c] > max_count:
                max_count = int(p.tokens[c])
                discard_color = c

        if max_count > 0:
            p.tokens[discard_color] -= 1
            state.gems_available[discard_color] += 1
            excess -= 1
            continue

        # Priority 3: only discard gold when all colours are at 0
        if p.tokens[Gem.GOLD] > 0:
            p.tokens[Gem.GOLD] -= 1
            state.gems_available[Gem.GOLD] += 1
            excess -= 1
            continue

        break  # Should not happen, but safe


# ---------------------------------------------------------------------------
# Game-end detection
# ---------------------------------------------------------------------------

def check_game_end(state: GameState) -> None:
    """Check and update game-over state after each action.

    Logic (2-player, equal-turns rule):
      - Uses turn_number parity to determine equal-turns status regardless
        of which player started first.
      - If a player reaches >= WINNING_POINTS and final_round_flag is NOT set:
          * turn_number even → just_acted has extra turn → opponent gets final turn.
          * turn_number odd  → equal turns already → game ends immediately.
      - If final_round_flag IS already set:
          * This means the extra turn just completed → game ends now.
    """
    p = state.players[state.current_player]
    just_acted = state.current_player

    if state.final_round_flag:
        # This was the extra turn for the opponent of the trigger-er
        state.game_over = True
        state.winner = _determine_winner(state)
        return

    if p.points >= WINNING_POINTS:
        # Equal-turns rule (2-player): use turn_number parity instead of
        # hardcoded player indices so the logic is correct regardless of
        # which player started first.
        #   turn_number even → just_acted has one more turn than opponent
        #       → opponent gets a final turn.
        #   turn_number odd  → both have had equal turns → game ends now.
        if state.turn_number % 2 == 0:
            # The player who just acted has an extra turn; opponent gets one more
            state.final_round_flag = True
            state.final_round_player = just_acted
        else:
            # Equal turns already — game ends immediately
            state.game_over = True
            state.winner = _determine_winner(state)


def _determine_winner(state: GameState) -> Optional[int]:
    """Determine winner after game ends.

    Tiebreakers (in order):
        1. Higher points
        2. Fewer purchased cards
        3. Draw (None)

    Returns:
        0 (P0 wins), 1 (P1 wins), or None (draw).
    """
    p0 = state.players[0]
    p1 = state.players[1]

    if p0.points > p1.points:
        return 0
    elif p1.points > p0.points:
        return 1
    elif p0.card_count < p1.card_count:
        return 0
    elif p1.card_count < p0.card_count:
        return 1
    else:
        return None  # Draw


# ---------------------------------------------------------------------------
# Card location helpers (for action mask and observation)
# ---------------------------------------------------------------------------

def get_face_up_card(state: GameState, position: int) -> Optional[Card]:
    """Get face-up card at flat position (0-11), or None if empty."""
    level = (position // FACE_UP_PER_LEVEL) + 1
    pos = position % FACE_UP_PER_LEVEL
    face_up_list = state.face_up.get(level, [])
    if pos < len(face_up_list):
        return face_up_list[pos]
    return None


def has_face_up_card(state: GameState, position: int) -> bool:
    """Check if a face-up card exists at the given flat position."""
    return get_face_up_card(state, position) is not None
