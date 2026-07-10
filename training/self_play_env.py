"""Self-Play environment wrapper for Splendor.

Wraps SplendorEnv to auto-play the opponent's turns. Only the agent's
transitions are returned via step(). Opponent transitions are otherwise
discarded, except that if the opponent's move ends the game, its terminal
reward (negated, since reward.py's terminal value is zero-sum from the
mover's perspective) is substituted for the agent's own stale pre-opponent
reward — the game's real outcome is only observable from the last mover's
step. This allows seamless integration with SB3's standard learn() loop.

For SubprocVecEnv compatibility: pass opponent_model_path (string) rather
than a loaded model. Each worker loads the model independently on init.
"""

import io as _io
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
        opponent_policy_bytes: Optional[bytes] = None,
        agent_player_idx: int = 0,
        opponent_player_idx: Optional[int] = None,
        deterministic_opponent: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_turns: int = 200,
    ):
        """Initialise the self-play environment.

        Args:
            cards_path: Path to cards_data.xlsx.
            opponent_model_path: Path to an SB3 MaskablePPO checkpoint, or None
                                 for a random opponent.
            opponent_policy_bytes: Serialized policy module bytes (preferred —
                                   avoids loading optimizer/buffer in each worker).
                                   Takes precedence over opponent_model_path.
            agent_player_idx: Which player the training agent controls (0 or 1).
            opponent_player_idx: Which player the opponent controls.
                                 Defaults to 1 - agent_player_idx.
            deterministic_opponent: If True, opponent uses greedy policy.
            seed: Random seed for reproducibility.
            render_mode: Passed through to SplendorEnv.
            max_turns: Maximum turns before truncation (safety cap).
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
            max_turns=max_turns,
        )

        # Load opponent model if provided
        self.opponent = None
        self._opponent_path = None
        if opponent_policy_bytes is not None:
            # Lightweight path: deserialize only the policy module (no
            # optimizer state, no rollout buffer).  ~6 MB per worker vs
            # ~40 MB for a full MaskablePPO.load() — critical when 20
            # subprocess workers start simultaneously.
            import torch
            # Ensure pickle can resolve the custom feature extractor class
            # inside the spawned subprocess.
            from training.feature_extractor import SplendorFeatureExtractor  # noqa: F401
            self.opponent = torch.load(
                _io.BytesIO(opponent_policy_bytes), map_location="cpu",
                weights_only=False,
            )
            # The policy was serialised from a model loaded with device="cpu",
            # so self.opponent.device should already be "cpu".  In PyTorch ≥2.6
            # nn.Module.device is a read-only property, so we bypass it via
            # __dict__ if a fix-up is ever needed.
            dev = self.opponent.__dict__.get("device", None)
            if dev is not None and str(dev) != "cpu":
                self.opponent.__dict__["device"] = "cpu"
        elif opponent_model_path is not None and os.path.exists(opponent_model_path):
            from sb3_contrib import MaskablePPO
            # device="cpu" forces all tensors to CPU regardless of save device
            self.opponent = MaskablePPO.load(
                opponent_model_path, device="cpu"
            )
            self._opponent_path = opponent_model_path

        # Expose same spaces as inner env
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # For seeding
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def __getstate__(self):
        """Custom pickle: exclude non-picklable SB3 opponent model.

        SubprocVecEnv workers pickle the env when returning get_attr results.
        The opponent model contains torch modules which can't be pickled.
        After unpickle, the opponent stays None — it's only needed in the
        subprocess where the env was originally created.
        """
        state = self.__dict__.copy()
        state["opponent"] = None
        return state

    def action_masks(self):
        """Return current action mask for sb3-contrib compatibility.

        MaskablePPO calls this via env_method('action_masks') to get masks
        from all parallel environments during rollout collection.
        """
        if self.inner.state is not None:
            return get_action_mask(self.inner.state, self.inner.state.current_player)
        return np.ones(N_ACTIONS, dtype=bool)

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
            obs, info, _, _ = self._auto_play_opponent(obs, info)

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
        if not terminated and not truncated:
            obs, info, opp_reward, opp_ended = self._auto_play_opponent(obs, info)
            # Check if game ended during opponent's turn
            terminated = self.inner.state.game_over
            if opp_ended:
                # The opponent's move ended the game, so the real outcome
                # only exists on the opponent's side of that step — the
                # agent's own last `reward` above is just a stale dense
                # shaping value from before the opponent moved. reward.py's
                # terminal value is zero-sum from the mover's perspective
                # (+1/-1/0 for win/loss/draw), so negate it for the agent.
                reward = -opp_reward

        return obs, reward, terminated, truncated, info

    def _auto_play_opponent(self, obs, info):
        """Auto-play opponent turns until it's the agent's turn again.

        Returns (obs, info, last_reward, game_ended). `last_reward` is the
        reward returned by the opponent's final step and is only meaningful
        when `game_ended` is True (it's then the terminal outcome from the
        opponent's perspective).
        """
        last_reward = 0.0
        game_ended = False
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

            obs, last_reward, terminated, truncated, info = self.inner.step(
                int(opp_action)
            )

            if terminated or truncated:
                game_ended = True
                break

        return obs, info, last_reward, game_ended

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
    opponent_policy_bytes: Optional[bytes] = None,
    agent_player_idx: int = 0,
    rank: int = 0,
    max_turns: int = 200,
):
    """Factory function for SubprocVecEnv.

    Args:
        cards_path: Path to cards_data.xlsx.
        opponent_model_path: Path to opponent checkpoint (or None).
        opponent_policy_bytes: Pre-serialized policy module bytes (preferred).
        agent_player_idx: Agent's player index.
        rank: Environment rank (used for seeding).
        max_turns: Maximum turns before truncation.

    Returns:
        A callable that creates a SelfPlayEnv.
    """
    def _init():
        return SelfPlayEnv(
            cards_path=cards_path,
            opponent_model_path=opponent_model_path,
            opponent_policy_bytes=opponent_policy_bytes,
            agent_player_idx=agent_player_idx,
            seed=rank,
            max_turns=max_turns,
        )
    return _init
