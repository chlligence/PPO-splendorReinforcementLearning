"""Game state dataclasses for Splendor.

State objects are pure data containers. All game logic lives in rules.py.
This separation makes cloning, serialisation, and testing straightforward.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .card import Card
from .constants import INITIAL_GEMS, FACE_UP_PER_LEVEL


@dataclass
class PlayerState:
    """All state for one player.

    Attributes:
        tokens: Array shape (6,) — counts of [BLACK, WHITE, RED, BLUE, GREEN, GOLD].
        bonuses: Array shape (5,) — permanent gem discounts from purchased cards.
        reserved: Cards held in hand (face-down, max 3).
        purchased: All cards the player has bought.
        points: Sum of victory points from purchased cards.
        card_count: len(purchased), cached for tiebreak efficiency.
    """
    tokens: np.ndarray       # shape (6,), dtype int32
    bonuses: np.ndarray      # shape (5,), dtype int32
    reserved: List[Card] = field(default_factory=list)
    purchased: List[Card] = field(default_factory=list)
    points: int = 0
    card_count: int = 0


@dataclass
class GameState:
    """Complete mutable game state for a Splendor match.

    The face-up cards are stored as lists of Card | None (None = empty slot).
    Decks are lists of remaining cards (top = end of list for efficient pop).
    """
    # Board
    face_up: Dict[int, List[Optional[Card]]] = field(default_factory=dict)
    decks: Dict[int, List[Card]] = field(default_factory=dict)
    gems_available: np.ndarray = field(
        default_factory=lambda: INITIAL_GEMS.copy()
    )

    # Players
    players: Tuple[PlayerState, PlayerState] = field(
        default_factory=lambda: (PlayerState(), PlayerState())
    )

    # Turn management
    current_player: int = 0          # 0 or 1
    turn_number: int = 0
    final_round_flag: bool = False   # Someone reached 15+ points
    final_round_player: Optional[int] = None  # Who triggered the final round
    game_over: bool = False
    winner: Optional[int] = None     # 0, 1, or None for draw


def new_player_state() -> PlayerState:
    """Create a fresh player with zero tokens, bonuses, and empty hands."""
    return PlayerState(
        tokens=np.zeros(6, dtype=np.int32),
        bonuses=np.zeros(5, dtype=np.int32),
        reserved=[],
        purchased=[],
        points=0,
        card_count=0,
    )


def clone_player_state(ps: PlayerState) -> PlayerState:
    """Deep-copy a PlayerState (numpy arrays are copied, card lists are shared)."""
    return PlayerState(
        tokens=ps.tokens.copy(),
        bonuses=ps.bonuses.copy(),
        reserved=list(ps.reserved),     # shallow copy of list; Cards are immutable
        purchased=list(ps.purchased),
        points=ps.points,
        card_count=ps.card_count,
    )


def clone_state(state: GameState) -> GameState:
    """Deep-copy an entire GameState.

    Cards are immutable (frozen dataclass), so card references can be safely
    shared between the original and the clone.
    """
    new_face_up = {}
    for level in [1, 2, 3]:
        new_face_up[level] = list(state.face_up.get(level, []))

    new_decks = {}
    for level in [1, 2, 3]:
        new_decks[level] = list(state.decks.get(level, []))

    return GameState(
        face_up=new_face_up,
        decks=new_decks,
        gems_available=state.gems_available.copy(),
        players=(
            clone_player_state(state.players[0]),
            clone_player_state(state.players[1]),
        ),
        current_player=state.current_player,
        turn_number=state.turn_number,
        final_round_flag=state.final_round_flag,
        final_round_player=state.final_round_player,
        game_over=state.game_over,
        winner=state.winner,
    )
