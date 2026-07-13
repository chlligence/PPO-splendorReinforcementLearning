"""Opponent pool for self-play training.

Maintains a collection of historical model checkpoints and provides
smart opponent selection to balance exploitation (play vs similar skill)
and exploration (play vs diverse opponents).

Selection strategy:
  - random_action_prob: random-action opponent (robustness insurance)
  - latest_prob: latest opponent (most recent skill level)
  - uniform_prob: uniform random from pool (diversity)
  - remaining: ELO-softmax-weighted (prefer stronger opponents)
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
    win_rate_vs_prev: Optional[float] = None  # Win rate vs previous best when added (None for gen 0 — no prior opponent)
    elo_source: str = "baseline"       # "full_eval" | "incremental_vs_latest" | "baseline"
    win_rate_source: str = "baseline"  # "full_eval" | "cheap_vs_latest" | "baseline"


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
        uniform_prob: float = 0.10,
        random_action_prob: float = 0.05,
        elo_temperature: float = 50.0,
    ):
        """Initialise the opponent pool.

        The ELO-softmax branch gets the remaining probability:
            1 − random_action_prob − latest_prob − uniform_prob
        (typically ~0.35). There is no separate elo_prob parameter —
        it was always dead code (stored but never used in sampling).

        Args:
            max_size: Maximum number of historical checkpoints to keep.
            latest_prob: Probability of selecting the most recent opponent.
            uniform_prob: Probability of uniform random selection FROM THE POOL
                (not random actions — that is random_action_prob).
            random_action_prob: Probability of returning None (→ random-action
                opponent).  Keeps a small robustness check against truly random
                play, independent of the pool's checkpoint diversity.
            elo_temperature: Temperature for ELO-softmax (ELO-scale units).
                At T=50, a 10-point gap → exp(0.2)≈1.22× weight instead of
                exp(10)≈22026× — the 40% branch actually samples proportionally
                rather than always picking the highest-ELO entry.
        """
        self.entries: List[PoolEntry] = []
        self.max_size = max_size
        self.latest_prob = latest_prob
        self.uniform_prob = uniform_prob
        self.random_action_prob = random_action_prob
        self.elo_temperature = elo_temperature

    def add(self, entry: PoolEntry) -> None:
        """Add a new checkpoint to the pool.

        If the pool exceeds max_size, the lowest-ELO entry is removed.
        """
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            # Remove entry with lowest ELO (keep diversity by ELO spread).
            # Uses min()+remove() rather than sort()+pop(0) so insertion
            # order (== generation order) is never disturbed — callers
            # rely on that order via get_latest_entry().
            removed = min(self.entries, key=lambda e: e.elo)
            self.entries.remove(removed)
            print(f"  [Pool] Removed gen {removed.generation} "
                  f"(ELO: {removed.elo:.0f}) — pool at {len(self.entries)}")

    def sample(
        self, rng: np.random.Generator
    ) -> Optional[PoolEntry]:
        """Sample an opponent from the pool.

        Returns None if the pool is empty, or when the random_action branch
        is selected (caller should use a random-action agent).

        Args:
            rng: NumPy random generator for reproducibility.

        Returns:
            A PoolEntry or None.
        """
        if len(self.entries) == 0:
            return None

        roll = rng.random()
        cursor = 0.0

        # Random-action opponent (truly random, not a pool checkpoint)
        cursor += self.random_action_prob
        if roll < cursor:
            return None

        # Latest opponent
        cursor += self.latest_prob
        if roll < cursor:
            return self.get_latest_entry()

        # Uniform random from pool (diversity)
        cursor += self.uniform_prob
        if roll < cursor:
            idx = rng.integers(0, len(self.entries))
            return self.entries[idx]

        # ELO-softmax-weighted sampling with temperature (remaining probability)
        # Without temperature, exp(raw ELO diff) makes a 10-point gap
        # → e^10 ≈ 22026× weight, effectively deterministic (always picks
        # the highest-ELO entry).  With T=50, the same gap → e^0.2 ≈ 1.22×.
        elos = np.array([e.elo for e in self.entries], dtype=np.float64)
        elos_centered = (elos - elos.max()) / self.elo_temperature
        probs = np.exp(elos_centered)
        probs /= probs.sum()
        idx = rng.choice(len(self.entries), p=probs)
        return self.entries[idx]

    def get_best_elo(self) -> float:
        """Return the highest ELO in the pool, or 1200 if empty."""
        if not self.entries:
            return 1200.0
        return max(e.elo for e in self.entries)

    def get_latest_entry(self) -> Optional[PoolEntry]:
        """Return the entry with the highest generation number, or None if empty.

        Determined explicitly by `generation`, never by list position —
        `self.entries` order is not a reliable proxy for recency once
        eviction or any other reordering has occurred.
        """
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.generation)

    def get_max_generation(self) -> Optional[int]:
        """Return the highest generation number in the pool, or None if empty."""
        if not self.entries:
            return None
        return max(e.generation for e in self.entries)

    def remove_generation(self, generation: int) -> None:
        """Remove all entries for a given generation (deduplication)."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.generation != generation]
        if len(self.entries) < before:
            print(f"  [Pool] Removed {before - len(self.entries)} duplicate(s) "
                  f"for generation {generation}")

    def size(self) -> int:
        """Number of entries in the pool."""
        return len(self.entries)

    # Format version for pool_index.json.  Bump when the serialisation format
    # or the semantic meaning of entries changes in a breaking way.
    POOL_INDEX_FORMAT_VERSION = 2  # v2: dict wrapper with version marker
                                   # v1 (pre-2026-07-12): bare list, agent-always-P0

    def save_index(self, path: str) -> None:
        """Save the pool index as JSON.

        Writes a dict with a format_version marker so load_index can detect
        and reject pools from incompatible training eras.

        Args:
            path: Path to save the JSON file (e.g., 'checkpoints/pool_index.json').
        """
        entries_data = []
        for e in self.entries:
            entries_data.append({
                "path": e.path,
                "generation": e.generation,
                "elo": e.elo,
                "win_rate_vs_prev": e.win_rate_vs_prev,
                "elo_source": e.elo_source,
                "win_rate_source": e.win_rate_source,
            })
        with open(path, "w") as f:
            json.dump({
                "format_version": self.POOL_INDEX_FORMAT_VERSION,
                "entries": entries_data,
            }, f, indent=2)

    def load_index(self, path: str, base_dir: str) -> None:
        """Load the pool index from a JSON file.

        Refuses to load pre-v2 (bare-list) pools — those were trained under
        the old "agent always P0, turn-flag always 0" distribution and are
        incompatible with the seat-asymmetry fix.

        Args:
            path: Path to read the JSON file from.
            base_dir: Base directory for resolving relative paths.
        """
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            data = json.load(f)

        # ---- Format detection ----
        if isinstance(data, list):
            # Pre-v2 format: bare list, no version marker.  These pools were
            # created before the seat-asymmetry fix and are incompatible.
            raise RuntimeError(
                f"\n{'!' * 60}\n"
                f"pool_index.json is in the OLD (pre-seat-fix) format.\n"
                f"These checkpoints were trained under 'always P0, turn-flag=0'\n"
                f"and WILL corrupt training if loaded into the new code.\n\n"
                f"ACTION: delete the checkpoints/ and logs/ directories, then\n"
                f"start a fresh training run.\n"
                f"{'!' * 60}"
            )
        if not isinstance(data, dict) or "format_version" not in data:
            raise RuntimeError(
                f"pool_index.json has an unrecognised format. "
                f"Delete checkpoints/ and start fresh."
            )
        version = data["format_version"]
        if version > self.POOL_INDEX_FORMAT_VERSION:
            raise RuntimeError(
                f"pool_index.json format_version={version} is newer than "
                f"this code supports (max {self.POOL_INDEX_FORMAT_VERSION}). "
                f"Update the training code or delete checkpoints/."
            )

        entries_data = data.get("entries", [])
        self.entries = []
        for d in entries_data:
            # Resolve relative paths
            checkpoint_path = d["path"]
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(base_dir, checkpoint_path)
            self.entries.append(PoolEntry(
                path=checkpoint_path,
                generation=d["generation"],
                elo=d["elo"],
                win_rate_vs_prev=d.get("win_rate_vs_prev"),
                elo_source=d.get("elo_source", "baseline"),
                win_rate_source=d.get("win_rate_source", "baseline"),
            ))
        print(f"  [Pool] Loaded {len(self.entries)} entries from {path}"
              f" (format v{version})")
