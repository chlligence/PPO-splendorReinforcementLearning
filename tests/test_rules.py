"""Unit tests for Splendor game rules.

Run with: python -m pytest tests/test_rules.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from splendor.constants import Gem, NUM_COLORS, COMBO_3_DIFFERENT
from splendor.card import Card, load_cards
from splendor.game_state import GameState, PlayerState, new_player_state, clone_state
from splendor.rules import (
    execute_action, check_game_end, can_afford,
)


def _make_card(card_id=0, level=1, bonus=Gem.BLUE, points=0,
               cost=(2, 1, 0, 0, 0)) -> Card:
    """Helper: create a test card."""
    return Card(card_id=card_id, level=level, bonus=bonus,
                points=points, cost=cost)


def _make_state() -> GameState:
    """Helper: create a minimal game state for testing."""
    face_up = {
        1: [_make_card(0, 1, Gem.BLUE, 0, (2, 1, 0, 0, 0)),
            _make_card(1, 1, Gem.RED, 0, (1, 2, 0, 0, 0)),
            _make_card(2, 1, Gem.GREEN, 1, (0, 0, 3, 0, 0)),
            _make_card(3, 1, Gem.BLACK, 0, (0, 0, 0, 2, 1))],
        2: [_make_card(4, 2, Gem.WHITE, 2, (0, 0, 3, 2, 2)),
            _make_card(5, 2, Gem.BLUE, 1, (3, 0, 0, 2, 2)),
            _make_card(6, 2, Gem.RED, 2, (0, 3, 0, 0, 3)),
            _make_card(7, 2, Gem.GREEN, 3, (0, 0, 0, 6, 0))],
        3: [_make_card(8, 3, Gem.BLUE, 3, (5, 3, 3, 0, 3)),
            _make_card(9, 3, Gem.BLACK, 4, (3, 0, 0, 3, 6)),
            _make_card(10, 3, Gem.WHITE, 4, (0, 0, 7, 0, 0)),
            _make_card(11, 3, Gem.RED, 5, (0, 7, 0, 3, 0))],
    }
    decks = {1: [], 2: [], 3: []}
    return GameState(
        face_up=face_up, decks=decks,
        gems_available=np.array([4, 4, 4, 4, 4, 5], dtype=np.int32),
        players=(new_player_state(), new_player_state()),
    )


class TestCanAfford:
    """Tests for affordability checking."""

    def test_simple_afford(self):
        """Can afford when tokens exactly cover cost."""
        tokens = np.array([2, 1, 0, 0, 0, 0], dtype=np.int32)  # 2 black, 1 white
        bonuses = np.zeros(5, dtype=np.int32)
        cost = (2, 1, 0, 0, 0)
        assert can_afford(tokens, bonuses, cost) is True

    def test_cannot_afford(self):
        """Cannot afford when tokens insufficient."""
        tokens = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        bonuses = np.zeros(5, dtype=np.int32)
        cost = (2, 1, 0, 0, 0)
        assert can_afford(tokens, bonuses, cost) is False

    def test_bonus_discount(self):
        """Bonuses reduce effective cost."""
        tokens = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        bonuses = np.array([1, 1, 0, 0, 0], dtype=np.int32)  # 1 black, 1 white bonus
        cost = (2, 1, 0, 0, 0)  # effective: (1, 0, 0, 0, 0)
        assert can_afford(tokens, bonuses, cost) is True

    def test_gold_wildcard(self):
        """Gold can substitute for any missing colour."""
        tokens = np.array([0, 0, 0, 0, 0, 2], dtype=np.int32)  # 2 gold
        bonuses = np.zeros(5, dtype=np.int32)
        cost = (1, 1, 0, 0, 0)
        assert can_afford(tokens, bonuses, cost) is True

    def test_gold_combined(self):
        """Gold + coloured tokens = affordable."""
        tokens = np.array([1, 0, 0, 0, 0, 1], dtype=np.int32)
        bonuses = np.zeros(5, dtype=np.int32)
        cost = (2, 1, 0, 0, 0)
        assert can_afford(tokens, bonuses, cost) is False  # need 3 total, have 2

    def test_expensive_card(self):
        """High-cost card with bonuses."""
        tokens = np.array([3, 3, 0, 0, 0, 2], dtype=np.int32)
        bonuses = np.array([2, 1, 0, 0, 0], dtype=np.int32)
        cost = (5, 3, 3, 0, 3)  # L3 card
        # Effective: (3, 2, 3, 0, 3) = 11 deficit, have 3+3=6 coloured + 2 gold = 8
        assert can_afford(tokens, bonuses, cost) is False


class TestBuyCard:
    """Tests for card purchase mechanics."""

    def test_buy_reduces_tokens(self):
        """Buying a card should correctly deduct tokens."""
        state = _make_state()
        p = state.players[0]
        p.tokens = np.array([2, 1, 0, 0, 0, 0], dtype=np.int32)  # 2 black, 1 white

        # Buy face-up card at position 0 (L1 pos0): cost (2,1,0,0,0)
        execute_action(state, 0, 30)  # Buy face-up pos 0

        # Should have paid 2 black, 1 white
        assert p.tokens[Gem.BLACK] == 0
        assert p.tokens[Gem.WHITE] == 0
        assert len(p.purchased) == 1
        assert p.bonuses[Gem.BLUE] == 1  # Card bonus was BLUE

    def test_buy_with_gold(self):
        """Gold should be used as wildcard."""
        state = _make_state()
        p = state.players[0]
        p.tokens = np.array([0, 0, 0, 0, 0, 3], dtype=np.int32)  # 3 gold

        # Buy card with cost (2,1,0,0,0) using gold
        execute_action(state, 0, 30)
        assert p.tokens[Gem.GOLD] == 0  # Used all 3 gold
        assert len(p.purchased) == 1

    def test_buy_with_bonuses(self):
        """Bonuses should discount the cost."""
        state = _make_state()
        p = state.players[0]
        p.tokens = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        p.bonuses = np.array([1, 1, 0, 0, 0], dtype=np.int32)  # 1 black, 1 white

        # Buy card cost (2,1,0,0,0) — effective (1,0,0,0,0)
        execute_action(state, 0, 30)
        assert p.tokens[Gem.BLACK] == 0
        assert len(p.purchased) == 1


class TestTakeGems:
    """Tests for gem-taking actions."""

    def test_take_3_different(self):
        """Take 3 different gems."""
        state = _make_state()
        p = state.players[0]

        # combo (0,1,2) = Black, White, Red
        execute_action(state, 0, 0)

        assert p.tokens[Gem.BLACK] == 1
        assert p.tokens[Gem.WHITE] == 1
        assert p.tokens[Gem.RED] == 1
        assert state.gems_available[Gem.BLACK] == 3
        assert state.gems_available[Gem.WHITE] == 3
        assert state.gems_available[Gem.RED] == 3

    def test_take_2_same(self):
        """Take 2 identical gems."""
        state = _make_state()
        p = state.players[0]
        # Black: 4 available -> valid
        execute_action(state, 0, 10)  # Take 2 Black
        assert p.tokens[Gem.BLACK] == 2
        assert state.gems_available[Gem.BLACK] == 2


class TestReserve:
    """Tests for card reservation."""

    def test_reserve_face_up_gives_gold(self):
        """Reserving from face-up should grant 1 gold if available."""
        state = _make_state()
        p = state.players[0]

        execute_action(state, 0, 15)  # Reserve L1 pos0

        assert len(p.reserved) == 1
        assert p.tokens[Gem.GOLD] == 1
        assert state.gems_available[Gem.GOLD] == 4

    def test_reserve_no_gold_if_empty(self):
        """Reserving when gold is exhausted should still work."""
        state = _make_state()
        state.gems_available[Gem.GOLD] = 0
        p = state.players[0]

        execute_action(state, 0, 15)

        assert len(p.reserved) == 1
        assert p.tokens[Gem.GOLD] == 0  # No gold granted

    def test_cannot_reserve_when_full(self):
        """Reserve should not be allowed when hand is full."""
        state = _make_state()
        p = state.players[0]
        p.reserved = [_make_card(99, 1, Gem.BLUE, 0) for _ in range(3)]

        # Action mask would prevent this — but the action would still
        # be masked. We test that the mask logic catches it.
        from splendor.action_mask import get_action_mask
        mask = get_action_mask(state, 0)
        # All reserve actions should be masked
        assert not any(mask[15:30])


class TestGameEnd:
    """Tests for game-end / final-round logic."""

    def test_p0_triggers_final_round(self):
        """P0 reaches 15: P1 gets one more turn."""
        state = _make_state()
        state.players[0].points = 14
        state.current_player = 0

        # P0 buys a 1-point card to reach 15
        # card at L1 pos2 has 1 point
        state.players[0].tokens = np.array([0, 0, 3, 0, 0, 0], dtype=np.int32)
        execute_action(state, 0, 32)  # Buy L1 pos2 (cost 3 green)
        check_game_end(state)

        assert state.final_round_flag is True
        assert state.final_round_player == 0
        assert state.game_over is False  # P1 still gets a turn

    def test_p1_triggers_game_ends(self):
        """P1 reaches 15: game ends immediately.

        In the normal game flow, P0 acts first (turn_number=0), then
        turn_number is incremented to 1 before P1 acts.  With turn_number=1
        (odd), the equal-turns rule triggers immediate game end because both
        players have had an equal number of turns at that point.
        """
        state = _make_state()
        state.players[1].points = 14
        state.current_player = 1
        state.turn_number = 1  # P0 has already taken their first turn

        # Give P1 tokens and buy the 1-point card
        state.players[1].tokens = np.array([0, 0, 3, 0, 0, 0], dtype=np.int32)
        execute_action(state, 1, 32)
        check_game_end(state)

        assert state.game_over is True
        assert state.winner == 1

    def test_tiebreak_fewer_cards(self):
        """Tie on points → fewer cards wins."""
        state = _make_state()
        state.players[0].points = 15
        state.players[0].card_count = 10
        state.players[1].points = 15
        state.players[1].card_count = 8
        # Force game over
        state.final_round_flag = True
        state.game_over = False

        # Simulate check after P1's extra turn
        check_game_end(state)
        # This won't trigger since no one just reached 15... let's just test
        # the winner determination directly
        from splendor.rules import _determine_winner
        state.game_over = True
        winner = _determine_winner(state)
        assert winner == 1  # P1 has fewer cards


class TestTokenOverflow:
    """Tests for automatic token discard."""

    def test_discard_on_overflow(self):
        """Taking gems that push over 10 should auto-discard."""
        state = _make_state()
        p = state.players[0]
        p.tokens = np.array([3, 3, 3, 0, 0, 1], dtype=np.int32)  # 10 total

        # Take 3 more — should trigger discard
        execute_action(state, 0, 0)  # Take Black,White,Red → total = 13

        total = int(np.sum(p.tokens))
        assert total <= 10, f"Expected ≤10 tokens, got {total}"


class TestCloneState:
    """Tests for state cloning."""

    def test_clone_is_independent(self):
        """Modifying clone should not affect original."""
        state = _make_state()
        cloned = clone_state(state)

        cloned.players[0].tokens[Gem.BLACK] = 5
        assert state.players[0].tokens[Gem.BLACK] == 0  # Original unchanged

    def test_clone_deep_copies_arrays(self):
        """Token and gems arrays should be independent."""
        state = _make_state()
        cloned = clone_state(state)

        cloned.gems_available[Gem.BLACK] = 0
        assert state.gems_available[Gem.BLACK] == 4  # Original unchanged
