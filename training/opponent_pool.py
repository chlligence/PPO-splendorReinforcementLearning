"""Opponent pool for self-play training.

Maintains a collection of historical model checkpoints and provides
smart opponent selection to balance exploitation (play vs similar skill)
and exploration (play vs diverse opponents).

Selection strategy:
  - 50% chance: latest opponent (most recent skill level)
  - 40% chance: ELO-softmax-weighted (prefer stronger opponents)
  - 10% chance: uniform random (diversity / avoid pool collapse)
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class PoolEntry:
    """One entry in the opponent pool."""
    path: str               # Checkpoint file path
    generation: int         # Generation number when created
    elo: float = 1200.0     # Estimated ELO rating
    win_rate_vs_prev: float = 0.5  # Win rate vs previous best when added


class OpponentPool:
    """Manages a collection of historical checkpoints for self-play.

    Usage:
        pool = OpponentPool(max_size=20)
        pool.add(PoolEntry(path="checkpoints/agent_gen_0.zip", generation=0))
        opponent = pool.sample(rng)  # Returns PoolEntry or None
    """

    def __init__(
        self,
        max_size: int = 20,
        latest_prob: float = 0.50,
        elo_prob: float = 0.40,
        random_prob: float = 0.10,
    ):
        """Initialise the opponent pool.

        Args:
            max_size: Maximum number of historical checkpoints to keep.
            latest_prob: Probability of selecting the most recent opponent.
            elo_prob: Probability of selecting by ELO-weighted sampling.
            random_prob: Probability of uniform random selection.
        """
        self.entries: List[PoolEntry] = []
        self.max_size = max_size
        self.latest_prob = latest_prob
        self.elo_prob = elo_prob
        self.random_prob = random_prob

    def add(self, entry: PoolEntry) -> None:
        """Add a new checkpoint to the pool.

        If the pool exceeds max_size, the lowest-ELO entry is removed.
        """
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            # Remove entry with lowest ELO (keep diversity by ELO spread)
            self.entries.sort(key=lambda e: e.elo)
            removed = self.entries.pop(0)
            print(f"  [Pool] Removed gen {removed.generation} "
                  f"(ELO: {removed.elo:.0f}) — pool at {len(self.entries)}")

    def sample(
        self, rng: np.random.Generator
    ) -> Optional[PoolEntry]:
        """Sample an opponent from the pool.

        Returns None if the pool is empty (caller should use a random agent).

        Args:
            rng: NumPy random generator for reproducibility.

        Returns:
            A PoolEntry or None.
        """
        if len(self.entries) == 0:
            return None

        roll = rng.random()

        if roll < self.latest_prob:
            # Return the most recent entry
            return self.entries[-1]
        elif roll < self.latest_prob + self.random_prob:
            # Uniform random
            idx = rng.integers(0, len(self.entries))
            return self.entries[idx]
        else:
            # ELO-softmax-weighted sampling
            elos = np.array([e.elo for e in self.entries], dtype=np.float64)
            # Softmax with temperature
            elos_centered = elos - elos.max()
            probs = np.exp(elos_centered)
            probs /= probs.sum()
            idx = rng.choice(len(self.entries), p=probs)
            return self.entries[idx]

    def get_best_elo(self) -> float:
        """Return the highest ELO in the pool, or 1200 if empty."""
        if not self.entries:
            return 1200.0
        return max(e.elo for e in self.entries)

    def size(self) -> int:
        """Number of entries in the pool."""
        return len(self.entries)

    def save_index(self, path: str) -> None:
        """Save the pool index as JSON.

        Args:
            path: Path to save the JSON file (e.g., 'checkpoints/pool_index.json').
        """
        data = []
        for e in self.entries:
            data.append({
                "path": e.path,
                "generation": e.generation,
                "elo": e.elo,
                "win_rate_vs_prev": e.win_rate_vs_prev,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_index(self, path: str, base_dir: str) -> None:
        """Load the pool index from a JSON file.

        Args:
            path: Path to read the JSON file from.
            base_dir: Base directory for resolving relative paths.
        """
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.entries = []
        for d in data:
            # Resolve relative paths
            checkpoint_path = d["path"]
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(base_dir, checkpoint_path)
            self.entries.append(PoolEntry(
                path=checkpoint_path,
                generation=d["generation"],
                elo=d["elo"],
                win_rate_vs_prev=d["win_rate_vs_prev"],
            ))
        print(f"  [Pool] Loaded {len(self.entries)} entries from {path}")
