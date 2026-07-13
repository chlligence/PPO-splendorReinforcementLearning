"""Gymnasium environment for the 2-player Splendor board game.

Implements the standard Gymnasium Env API with action masking support
for Stable-Baselines3's MaskablePPO.

Key design decisions:
  - Observation is always from the perspective of the player about to act.
  - Action mask is returned in the info dict as "action_mask".
  - Token overflow (>10) is handled automatically with a deterministic
    discard heuristic (see rules._auto_discard_if_needed).
  - The opponent's reserved card contents are hidden (imperfect information).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .constants import (
    CARDS_PER_LEVEL, FACE_UP_PER_LEVEL,
)
from .card import Card, load_cards
from .game_state import (
    GameState, PlayerState,
    new_player_state, clone_state,
)
from .rules import execute_action, check_game_end, can_afford
from .action_mask import get_action_mask, get_action_description, N_ACTIONS
from .observation import build_observation, OBS_DIM
from .rewardcc import compute_reward


class SplendorEnv(gym.Env):
    """Gymnasium environment for 2-player Splendor.

    Observation space: Box(203,) — flat float32 vector, all features in [0, 1].
    Action space: Discrete(51) — with action masking via info["action_mask"].

    Usage:
        cards = load_cards("data/cards_data.xlsx")
        env = SplendorEnv(cards)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {
        "render_modes": ["ansi", "human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        cards_by_level: Dict[int, List[Card]],
        render_mode: Optional[str] = None,
        starting_player: int = 0,
        max_turns: int = 200,
        shaping_gamma: float = 0.99,
    ):
        """Initialise the environment.

        Args:
            cards_by_level: Card database from load_cards().
            render_mode: "ansi", "human", or None.
            starting_player: Which player goes first (0 or 1).
            max_turns: Maximum turns before truncation (safety cap).
            shaping_gamma: Discount factor for PBRS shaping. MUST equal the
                PPO trainer's gamma for the policy-invariance guarantee.
        """
        super().__init__()
        self.cards_by_level = cards_by_level
        self.render_mode = render_mode
        self.starting_player = starting_player
        self.max_turns = max_turns
        self.shaping_gamma = shaping_gamma

        # ---- Observation space ----
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # ---- Action space ----
        self.action_space = spaces.Discrete(N_ACTIONS)

        # ---- Internal state ----
        self.state: Optional[GameState] = None
        self.rng = np.random.default_rng()

    def action_masks(self):
        """Return current action mask for sb3-contrib compatibility."""
        if self.state is not None:
            return get_action_mask(self.state, self.state.current_player)
        return np.zeros(N_ACTIONS, dtype=bool)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment to start a new game.

        Returns:
            (observation, info) where info contains "action_mask".
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. Shuffle decks and deal face-up cards
        decks: Dict[int, List[Card]] = {}
        face_up: Dict[int, List[Optional[Card]]] = {}

        for level in [1, 2, 3]:
            cards = list(self.cards_by_level[level])
            self.rng.shuffle(cards)
            # First 4 cards face-up, rest in deck
            face_up[level] = cards[:FACE_UP_PER_LEVEL]
            decks[level] = cards[FACE_UP_PER_LEVEL:]

        # 2. Initialise game state
        self.state = GameState(
            face_up=face_up,
            decks=decks,
            gems_available=np.array([4, 4, 4, 4, 4, 5], dtype=np.int32),
            players=(
                new_player_state(),
                new_player_state(),
            ),
            current_player=self.starting_player,
            turn_number=0,
            final_round_flag=False,
            final_round_player=None,
            game_over=False,
            winner=None,
        )

        obs = build_observation(self.state, self.state.current_player)
        mask = get_action_mask(self.state, self.state.current_player)
        return obs, {"action_mask": mask}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one action and advance the game.

        Args:
            action: Discrete action index (0–50).

        Returns:
            (observation, reward, terminated, truncated, info)
            info contains "action_mask" for the next player.
        """
        # ---- Pre-action snapshot for reward computation ----
        prev_state = clone_state(self.state)
        player_idx = self.state.current_player

        # ---- Execute action ----
        execute_action(self.state, player_idx, action)

        # ---- Check natural game end (15-pt trigger / final round) ----
        check_game_end(self.state)

        # ---- Switch turns (if game continues) ----
        if not self.state.game_over:
            self.state.current_player = 1 - self.state.current_player
            self.state.turn_number += 1

        # ---- Force termination when turn limit exceeded ----
        # Must happen BEFORE compute_reward so the ±10 terminal reward is
        # correctly issued for truncated games (previously the truncation
        # happened after reward computation, so games hitting the 200-turn
        # cap only received dense shaping, never the win/loss outcome).
        truncated = self.state.turn_number >= self.max_turns
        if truncated and not self.state.game_over:
            from .rules import _determine_winner
            self.state.game_over = True
            self.state.winner = _determine_winner(self.state)

        # ---- Compute reward for the player who just acted ----
        # Pass shaping_gamma explicitly so the PBRS term uses the SAME γ as
        # the PPO trainer — the invariance guarantee depends on this match.
        reward = compute_reward(prev_state, self.state, action, player_idx,
                                gamma=self.shaping_gamma)

        # ---- Build return values ----
        terminated = self.state.game_over

        # Observation for the NEXT player (or terminal state)
        next_player = self.state.current_player if not terminated else player_idx
        obs = build_observation(self.state, next_player)

        # Action mask for the next player (empty if game over)
        if not terminated:
            mask = get_action_mask(self.state, next_player)
        else:
            mask = np.zeros(N_ACTIONS, dtype=bool)

        return obs, float(reward), terminated, truncated, {"action_mask": mask}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        """Render the current game state as text."""
        if self.render_mode is None:
            return None

        state = self.state
        color_names = ["Black", "White", "Red", "Blue", "Green"]

        lines = []
        lines.append("=" * 60)
        lines.append(f"Turn: {state.turn_number}  |  "
                      f"Current Player: {state.current_player}"
                      + (" [FINAL ROUND]" if state.final_round_flag else ""))

        # Board gems
        gem_strs = []
        for c in range(5):
            gem_strs.append(f"{color_names[c]}:{state.gems_available[c]}")
        gem_strs.append(f"Gold:{state.gems_available[5]}")
        lines.append("Gems: " + " | ".join(gem_strs))

        # Face-up cards
        lines.append("-" * 60)
        for level in [1, 2, 3]:
            lines.append(f"--- Level {level} (deck: {len(state.decks[level])}) ---")
            for pos, card in enumerate(state.face_up.get(level, [])):
                if card is not None:
                    cost_str = " ".join(
                        f"{color_names[c][:1]}:{card.cost[c]}"
                        for c in range(5) if card.cost[c] > 0
                    )
                    lines.append(
                        f"  [{pos}] {card.bonus.name} +{card.points}pt  "
                        f"Cost: {cost_str}"
                    )
                else:
                    lines.append(f"  [{pos}] (empty)")

        # Players
        for i in [0, 1]:
            p = state.players[i]
            marker = "← CURRENT" if i == state.current_player else ""
            lines.append("-" * 60)
            lines.append(f"Player {i} {marker}")
            lines.append(f"  Points: {p.points}  |  Cards: {p.card_count}")

            token_str = " ".join(
                f"{color_names[c][:1]}:{p.tokens[c]}"
                for c in range(6)
            )
            lines.append(f"  Tokens: {token_str} (total: {int(np.sum(p.tokens))})")

            bonus_str = " ".join(
                f"{color_names[c][:1]}:{p.bonuses[c]}"
                for c in range(5)
            )
            lines.append(f"  Bonuses: {bonus_str}")
            lines.append(f"  Reserved: {len(p.reserved)} cards")

        if state.game_over:
            lines.append("=" * 60)
            if state.winner is not None:
                lines.append(f"WINNER: Player {state.winner}")
            else:
                lines.append("RESULT: Draw")

        lines.append("=" * 60)
        output = "\n".join(lines)

        if self.render_mode == "ansi":
            return output
        elif self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self):
        """Clean up resources."""
        pass
