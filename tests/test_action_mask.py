"""Tests for action mask computation.

Run with: python -m pytest tests/test_action_mask.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from splendor.constants import Gem
from splendor.card import Card
from splendor.game_state import GameState, PlayerState, new_player_state
from splendor.action_mask import get_action_mask, N_ACTIONS


def _card(card_id=0, level=1, bonus=Gem.BLUE, points=0,
          cost=(2, 1, 0, 0, 0)) -> Card:
    return Card(card_id=card_id, level=level, bonus=bonus,
                points=points, cost=cost)


def _state() -> GameState:
    face_up = {
        1: [_card(0, 1, Gem.BLUE, 0, (2, 1, 0, 0, 0)),
            _card(1, 1, Gem.RED, 0, (1, 2, 0, 0, 0)),
            _card(2, 1, Gem.GREEN, 1, (0, 0, 3, 0, 0)),
            _card(3, 1, Gem.BLACK, 0, (0, 0, 0, 2, 1))],
        2: [_card(4, 2, Gem.WHITE, 2, (0, 0, 3, 2, 2)),
            _card(5, 2, Gem.BLUE, 1, (3, 0, 0, 2, 2)),
            _card(6, 2, Gem.RED, 2, (0, 3, 0, 0, 3)),
            _card(7, 2, Gem.GREEN, 3, (0, 0, 0, 6, 0))],
        3: [_card(8, 3, Gem.BLUE, 3, (5, 3, 3, 0, 3)),
            _card(9, 3, Gem.BLACK, 4, (3, 0, 0, 3, 6)),
            _card(10, 3, Gem.WHITE, 4, (0, 0, 7, 0, 0)),
            _card(11, 3, Gem.RED, 5, (0, 7, 0, 3, 0))],
    }
    decks = {1: [_card(50+i, 1) for i in range(5)],
             2: [_card(60+i, 2) for i in range(5)],
             3: [_card(70+i, 3) for i in range(5)]}
    return GameState(
        face_up=face_up, decks=decks,
        gems_available=np.array([4, 4, 4, 4, 4, 5], dtype=np.int32),
        players=(new_player_state(), new_player_state()),
    )


class TestInitialMask:
    """Test action mask at the start of a game."""

    def test_all_take_3_legal(self):
        """All 10 take-3 combos should be legal (all colours have 4 gems)."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert all(mask[0:10]), "All take-3 combos should be legal"

    def test_all_take_2_legal(self):
        """All 5 take-2-same should be legal (all colours at 4)."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert all(mask[10:15]), "All take-2-same should be legal"

    def test_no_buy_legal_initially(self):
        """No buys should be legal with zero tokens."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert not any(mask[30:45]), "No buys should be legal with 0 tokens"

    def test_reserve_face_up_legal(self):
        """All face-up reserve actions should be legal initially."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert all(mask[15:27]), "All face-up reserves should be legal"

    def test_reserve_deck_legal(self):
        """All deck reserve actions should be legal when decks have cards."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert all(mask[27:30]), "All deck reserves should be legal"


class TestGemExhaustion:
    """Test mask when gems are depleted."""

    def test_take_2_blocked_at_3(self):
        """Take-2-same should be illegal when only 3 of that colour remain."""
        state = _state()
        state.gems_available[Gem.BLACK] = 3
        mask = get_action_mask(state, 0)
        assert not mask[10], "Take 2 Black should be blocked at 3"

    def test_take_2_ok_at_4(self):
        """Take-2-same should be legal at exactly 4."""
        state = _state()
        state.gems_available[Gem.BLACK] = 4
        mask = get_action_mask(state, 0)
        assert mask[10], "Take 2 Black should be legal at 4"

    def test_take_3_blocked_when_depleted(self):
        """Combos requiring a depleted colour should be blocked."""
        state = _state()
        state.gems_available[Gem.BLACK] = 0
        mask = get_action_mask(state, 0)
        # All combos involving Black (0) should be blocked
        for i, combo in enumerate([(0, 1, 2), (0, 1, 3), (0, 1, 4),
                                     (0, 2, 3), (0, 2, 4), (0, 3, 4)]):
            assert not mask[i], f"Combo {combo} should be blocked"


class TestReserveFull:
    """Test mask when player's hand is full."""

    def test_all_reserves_blocked_when_full(self):
        """All reserve actions blocked when holding 3 cards."""
        state = _state()
        state.players[0].reserved = [_card(99, 1), _card(98, 1), _card(97, 1)]
        mask = get_action_mask(state, 0)
        assert not any(mask[15:30]), "All reserves should be blocked when hand full"


class TestBuyMask:
    """Test mask for buy actions."""

    def test_buy_shows_when_affordable(self):
        """Buy actions visible when player has tokens."""
        state = _state()
        state.players[0].tokens = np.array([2, 1, 0, 0, 0, 0], dtype=np.int32)
        mask = get_action_mask(state, 0)
        # L1 pos0 costs (2,1,0,0,0) — should be affordable
        assert mask[30], "Should be able to buy L1 pos0"

    def test_buy_blocked_when_unaffordable(self):
        """Buy actions blocked when insufficient tokens."""
        state = _state()
        state.players[0].tokens = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        mask = get_action_mask(state, 0)
        assert not any(mask[30:45]), "No buys should be legal with 0 tokens"

    def test_buy_reserved(self):
        """Reserved card buy should appear when affordable."""
        state = _state()
        p = state.players[0]
        p.reserved = [_card(99, 1, Gem.BLUE, 0, (2, 1, 0, 0, 0))]
        p.tokens = np.array([2, 1, 0, 0, 0, 0], dtype=np.int32)
        mask = get_action_mask(state, 0)
        assert mask[42], "Should be able to buy reserved card"


class TestMaskShape:
    """Verify mask properties."""

    def test_mask_shape(self):
        """Mask should have exactly N_ACTIONS entries."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert len(mask) == N_ACTIONS
        assert mask.dtype == bool

    def test_at_least_one_legal(self):
        """There should always be at least one legal action (pass is always legal)."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert np.any(mask), "Should be at least one legal action"
        assert mask[50], "Pass action must always be legal"

    def test_take_1_actions(self):
        """Take-1-gem should be legal when gems are available."""
        state = _state()
        mask = get_action_mask(state, 0)
        assert all(mask[45:50]), "All take-1-gem should be legal initially"

        # Deplete all gems
        state.gems_available[:5] = 0
        mask = get_action_mask(state, 0)
        assert not any(mask[45:50]), "Take-1-gem blocked when gems at 0"
