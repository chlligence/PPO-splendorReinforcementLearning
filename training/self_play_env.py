"""Self-Play environment wrapper for Splendor.

Wraps SplendorEnv to auto-play the opponent's turns. Only the agent's
transitions are returned via step() — opponent transitions are discarded.
This allows seamless integration with SB3's standard learn() loop.

For SubprocVecEnv compatibility: pass opponent_model_path (string) rather
than a loaded model. Each worker loads the model independently on init.
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from splendor.env import SplendorEnv
from splendor.card import load_cards
from splendor.action_mask import get_action_mask, N_ACTIONS
from splendor.observation import build_observation, OBS_DIM


class SelfPlayEnv(gym.Env):
    """Wraps SplendorEnv for self-play: agent acts, opponent auto-plays.

    The agent always plays as player `agent_player_idx`. When the opponent's
    turn comes, the environment automatically queries the opponent model's
    policy and executes the chosen action.

    Only transitions where the AGENT acts are returned — opponent steps
    are consumed internally and their rewards discarded.

    Usage with SubprocVecEnv:
        def make_env(rank):
            return SelfPlayEnv(
                cards_path="cards_data.xlsx",
                opponent_model_path=None,  # or path to checkpoint
                agent_player_idx=0,
                seed=rank,
            )
        vec_env = SubprocVecEnv([make_env for i in range(20)])
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cards_path: str,
        opponent_model_path: Optional[str] = None,
        agent_player_idx: int = 0,
        opponent_player_idx: Optional[int] = None,
        deterministic_opponent: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialise the self-play environment.

        Args:
            cards_path: Path to cards_data.xlsx.
            opponent_model_path: Path to an SB3 MaskablePPO checkpoint, or None
                                 for a random opponent.
            agent_player_idx: Which player the training agent controls (0 or 1).
            opponent_player_idx: Which player the opponent controls.
                                 Defaults to 1 - agent_player_idx.
            deterministic_opponent: If True, opponent uses greedy policy.
            seed: Random seed for reproducibility.
            render_mode: Passed through to SplendorEnv.
        """
        super().__init__()
        self.agent_idx = agent_player_idx
        self.opponent_idx = (
            opponent_player_idx
            if opponent_player_idx is not None
            else 1 - agent_player_idx
        )
        self.deterministic_opponent = deterministic_opponent

        # Load card data
        self.cards = load_cards(cards_path)

        # Create inner environment — agent always starts as current_player at reset
        self.inner = SplendorEnv(
            self.cards,
            render_mode=render_mode,
            starting_player=agent_player_idx,
        )

        # Load opponent model if provided
        self.opponent = None
        if opponent_model_path is not None and os.path.exists(opponent_model_path):
            # Lazy import to avoid dependency issues if SB3 not installed
            from sb3_contrib import MaskablePPO
            self.opponent = MaskablePPO.load(opponent_model_path)
            self._opponent_path = opponent_model_path

        # Expose same spaces as inner env
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # For seeding
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        """Reset the game. Agent starts first.

        Returns the agent's initial observation and info dict.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obs, info = self.inner.reset(seed=seed)

        # If opponent goes first (rare — only if agent_player_idx != 0 config),
        # auto-play opponent's first turn
        if self.inner.state.current_player != self.agent_idx:
            obs, info = self._auto_play_opponent(obs, info)

        return obs, info

    def step(self, action: int):
        """Execute agent's action, then auto-play opponent until agent's turn.

        Args:
            action: The agent's chosen action (0-50).

        Returns:
            (obs, reward, terminated, truncated, info)
            reward is from the agent's perspective.
        """
        # Agent's action
        obs, reward, terminated, truncated, info = self.inner.step(action)

        # Auto-play opponent until it's agent's turn again or game ends
        if not terminated:
            obs, info = self._auto_play_opponent(obs, info)
            # Check if game ended during opponent's turn
            terminated = self.inner.state.game_over

        return obs, reward, terminated, truncated, info

    def _auto_play_opponent(self, obs, info):
        """Auto-play opponent turns until it's the agent's turn again.

        Returns the observation and info for the agent's next turn.
        """
        while (not self.inner.state.game_over
               and self.inner.state.current_player != self.agent_idx):
            opp_idx = self.inner.state.current_player
            opp_obs = build_observation(self.inner.state, opp_idx)
            opp_mask = get_action_mask(self.inner.state, opp_idx)

            if self.opponent is not None:
                opp_action, _ = self.opponent.predict(
                    opp_obs,
                    action_masks=opp_mask,
                    deterministic=self.deterministic_opponent,
                )
            else:
                # Random legal action
                legal = np.where(opp_mask)[0]
                if len(legal) == 0:
                    legal = np.array([50])  # fallback to pass
                opp_action = self.rng.choice(legal)

            obs, reward, terminated, truncated, info = self.inner.step(
                int(opp_action)
            )

            if terminated:
                break

        return obs, info

    def render(self):
        """Render the current game state."""
        return self.inner.render()

    def close(self):
        """Clean up resources."""
        if self.inner is not None:
            self.inner.close()


def make_env_fn(
    cards_path: str,
    opponent_model_path: Optional[str],
    agent_player_idx: int = 0,
    rank: int = 0,
):
    """Factory function for SubprocVecEnv.

    Args:
        cards_path: Path to cards_data.xlsx.
        opponent_model_path: Path to opponent checkpoint (or None).
        agent_player_idx: Agent's player index.
        rank: Environment rank (used for seeding).

    Returns:
        A callable that creates a SelfPlayEnv.
    """
    def _init():
        return SelfPlayEnv(
            cards_path=cards_path,
            opponent_model_path=opponent_model_path,
            agent_player_idx=agent_player_idx,
            seed=rank,
        )
    return _init
