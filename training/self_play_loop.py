"""Self-Play training loop for Splendor RL.

Orchestrates iterative training: each generation trains the agent against
a sampled opponent from the pool, then evaluates and adds the new policy
to the pool.

Uses Stable-Baselines3 MaskablePPO with SubprocVecEnv for parallelism.
"""

import math
import os
import time
from typing import Optional
import numpy as np

from splendor.card import load_cards

from .config import (
    PPO_CONFIG, NETWORK_CONFIG, SELFPLAY_CONFIG, ENV_CONFIG,
    CHECKPOINT_DIR, LOG_DIR, CARDS_PATH,
)
from .feature_extractor import SplendorFeatureExtractor
from .opponent_pool import OpponentPool, PoolEntry
from .self_play_env import make_env_fn, SelfPlayEnv
from .evaluate import evaluate_generation


def create_agent_model(
    vec_env,
    features_dim: int = 256,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    device: str = "cuda",
    tensorboard_log: Optional[str] = None,
):
    """Create a new MaskablePPO agent with SplendorFeatureExtractor.

    Args:
        vec_env: Vectorized environment.
        features_dim: Output dimension of the feature extractor.
        learning_rate: PPO learning rate.
        ent_coef: Entropy coefficient.
        device: "cuda" or "cpu".
        tensorboard_log: Path for TensorBoard logs.

    Returns:
        A MaskablePPO instance.
    """
    from sb3_contrib import MaskablePPO

    policy_kwargs = dict(
        features_extractor_class=SplendorFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(
            pi=NETWORK_CONFIG["pi_layers"],
            vf=NETWORK_CONFIG["vf_layers"],
        ),
    )

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        learning_rate=learning_rate,
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=ent_coef,
        vf_coef=PPO_CONFIG["vf_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=device,
    )
    return model


def compute_entropy_coef(generation: int) -> float:
    """Linearly anneal entropy coefficient over generations.

    Starts at ent_start (gen 0), ends at ent_end after ent_anneal_generations.
    """
    start = SELFPLAY_CONFIG["ent_start"]
    end = SELFPLAY_CONFIG["ent_end"]
    anneal = SELFPLAY_CONFIG["ent_anneal_generations"]

    if generation >= anneal:
        return end
    return start + (end - start) * (generation / anneal)


def compute_learning_rate(generation: int) -> float:
    """Cosine-anneal learning rate over generations.

    Starts at PPO_CONFIG learning_rate (gen 0), decays to 1e-5 by the final
    generation.  Cosine schedule keeps LR higher longer than linear decay,
    giving more time for exploration before settling into fine-tuning.
    """
    start: float = float(PPO_CONFIG["learning_rate"])  # 5e-5
    end: float = 1e-5
    total = SELFPLAY_CONFIG["generations"]
    if generation >= total:
        return end
    progress = generation / total  # 0.0 → ~1.0
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def run_self_play(
    cards_path: str = CARDS_PATH,
    resume_from: Optional[str] = None,
) -> None:
    """Run the full self-play training pipeline.

    Args:
        cards_path: Path to cards_data.xlsx.
        resume_from: Path to a checkpoint to resume from (optional).
    """
    # ---- pool_index.json versioning ----
    # OpponentPool.load_index() detects and rejects pre-seat-fix pools
    # (bare-list format = v1).  New pools are saved with format_version=2.
    # If you see a RuntimeError about old format, delete checkpoints/ and
    # start fresh — old checkpoints are incompatible with the new training
    # distribution.
    pool_index_path = os.path.join(CHECKPOINT_DIR, "pool_index.json")

    # ---- Config consistency checks ----
    # PBRS invariance requires the shaping discount to match the PPO trainer's
    # gamma exactly.  A mismatch silently breaks the telescoping-sum property.
    ppo_gamma = PPO_CONFIG["gamma"]
    env_gamma = ENV_CONFIG["shaping_gamma"]
    assert abs(ppo_gamma - env_gamma) < 1e-9, (
        f"PPO_CONFIG['gamma'] ({ppo_gamma}) != ENV_CONFIG['shaping_gamma'] "
        f"({env_gamma}) — PBRS invariance broken!"
    )

    print("=" * 60)
    print("Splendor Self-Play Training")
    print("=" * 60)
    print(f"Device: {PPO_CONFIG['device']}")
    print(f"Parallel envs: {SELFPLAY_CONFIG['n_envs']}")
    print(f"Steps per generation: {SELFPLAY_CONFIG['steps_per_generation']:,}")
    print(f"Generations: {SELFPLAY_CONFIG['generations']}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Log dir: {LOG_DIR}")

    # Load cards
    print(f"\nLoading cards from: {cards_path}")
    cards = load_cards(cards_path)
    print(f"Loaded {sum(len(v) for v in cards.values())} cards.")

    # Initialise or load pool
    pool = OpponentPool(
        max_size=SELFPLAY_CONFIG["opponent_pool_size"],
        latest_prob=SELFPLAY_CONFIG["latest_opponent_prob"],
        uniform_prob=SELFPLAY_CONFIG["random_opponent_prob"],  # uniform from pool
        random_action_prob=SELFPLAY_CONFIG["random_action_prob"],
        elo_temperature=SELFPLAY_CONFIG["elo_temperature"],
    )
    if os.path.exists(pool_index_path):
        pool.load_index(pool_index_path, CHECKPOINT_DIR)

    rng = np.random.default_rng()

    # Determine starting generation
    start_gen = 0
    agent_model = None
    latest_path = os.path.join(CHECKPOINT_DIR, "agent_latest.zip")  # fallback

    if resume_from and os.path.exists(resume_from):
        from sb3_contrib import MaskablePPO
        print(f"\nResuming from: {resume_from}")
        agent_model = MaskablePPO.load(resume_from)
        # Extract generation from filename
        import re
        match = re.search(r'gen_(\d+)', resume_from)
        if match:
            start_gen = int(match.group(1)) + 1
        print(f"Resuming from generation {start_gen}")

    # Running estimate of the agent's own ELO, carried forward across
    # generations so estimate_elo() never resets to an arbitrary baseline.
    # Best available proxy on resume is the pool's own latest recorded ELO.
    latest_pool_entry = pool.get_latest_entry()
    last_known_elo = latest_pool_entry.elo if latest_pool_entry is not None else 1200.0

    # ---- Create persistent vectorized environments (once for all generations) ----
    # On Windows (spawn mode), recreating SubprocVecEnv each generation costs
    # 15–40 s of process-spawn + import overhead per generation.  Workers persist
    # across generations; opponent weights are hot-swapped via env_method.
    from stable_baselines3.common.vec_env import SubprocVecEnv

    n_envs = SELFPLAY_CONFIG["n_envs"]
    print(f"\nCreating {n_envs} persistent parallel environments...")
    env_fns = [
        make_env_fn(
            cards_path=cards_path,
            opponent_model_path=None,      # hot-swap handles opponent assignment
            opponent_policy_bytes=None,    # start with random opponent
            agent_player_idx=i % 2,        # alternate P0/P1 so agent sees both seats
            rank=i,                        # envs persist — RNG states evolve naturally
            max_turns=ENV_CONFIG["max_turns"],
            shaping_gamma=ENV_CONFIG["shaping_gamma"],
        )
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    _agent_model_needs_set_env = True  # track first-time set_env

    try:
        # ---- Main training loop ----
        for generation in range(start_gen, SELFPLAY_CONFIG["generations"]):
            gen_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{SELFPLAY_CONFIG['generations']}")
            print(f"{'='*60}")

            # 1. Sample opponent
            opponent_entry = pool.sample(rng)
            opponent_path = opponent_entry.path if opponent_entry else None
            if opponent_entry:
                print(f"Opponent: gen {opponent_entry.generation} "
                      f"(ELO: {opponent_entry.elo:.0f})")
            else:
                print("Opponent: Random (no pool entries yet)")

            # 1a. Serialise opponent policy to bytes (lightweight — no optimizer
            #     state, no rollout buffer).  Deserialised in each worker via
            #     set_opponent() (~6 MB per worker vs ~40 MB for full load).
            opponent_policy_bytes: Optional[bytes] = None
            if opponent_path is not None:
                import io as _io
                import torch as _torch
                from sb3_contrib import MaskablePPO as _MaskablePPO
                _opp_full = _MaskablePPO.load(opponent_path, device="cpu")
                _buf = _io.BytesIO()
                _torch.save(_opp_full.policy, _buf)
                opponent_policy_bytes = _buf.getvalue()
                del _opp_full, _buf  # free memory before pushing to workers

            # 1b. Hot-swap opponent in all workers (no process respawn needed).
            vec_env.env_method("set_opponent", opponent_policy_bytes)

            # 2. Create or update agent model
            ent_coef = compute_entropy_coef(generation)
            print(f"Entropy coefficient: {ent_coef:.4f}")

            if agent_model is None:
                agent_model = create_agent_model(
                    vec_env,
                    features_dim=NETWORK_CONFIG["features_dim"],
                    learning_rate=compute_learning_rate(generation),
                    ent_coef=ent_coef,
                    device=PPO_CONFIG["device"],
                    tensorboard_log=LOG_DIR,
                )
                _agent_model_needs_set_env = False
            else:
                if _agent_model_needs_set_env:
                    agent_model.set_env(vec_env)
                    _agent_model_needs_set_env = False
                agent_model.ent_coef = ent_coef
                agent_model.learning_rate = compute_learning_rate(generation)
                agent_model._setup_lr_schedule()

            # 3. Train
            print(f"Training for {SELFPLAY_CONFIG['steps_per_generation']:,} steps...")
            agent_model.learn(
                total_timesteps=SELFPLAY_CONFIG["steps_per_generation"],
                reset_num_timesteps=False,
                tb_log_name=f"gen_{generation}",
            )

            # 4. Save checkpoint
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"agent_gen_{generation}.zip"
            )
            latest_path = os.path.join(CHECKPOINT_DIR, "agent_latest.zip")
            agent_model.save(checkpoint_path)
            agent_model.save(latest_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # 5. Evaluate (workers remain alive but idle — they block on
            #    SubprocVecEnv pipes with near-zero CPU usage during eval).
            eval_result = evaluate_generation(
                agent_model, pool, cards_path, generation, last_known_elo,
                eval_interval_generations=SELFPLAY_CONFIG["eval_interval_generations"],
                cheap_eval_games=SELFPLAY_CONFIG["cheap_eval_games"],
                cheap_elo_k=SELFPLAY_CONFIG["cheap_elo_k"],
                full_eval_games=max(20, SELFPLAY_CONFIG["eval_games"] // 2),
            )
            last_known_elo = eval_result["elo"]

            # 6. Add to pool (remove any existing entry for this generation first)
            pool.entries = [e for e in pool.entries if e.generation != generation]
            pool.add(PoolEntry(
                path=checkpoint_path,
                generation=generation,
                elo=eval_result["elo"],
                win_rate_vs_prev=eval_result["win_rate_vs_prev"],
                elo_source=eval_result["elo_source"],
                win_rate_source=eval_result["win_rate_source"],
            ))
            pool.save_index(pool_index_path)

            gen_time = time.time() - gen_start_time
            print(f"Generation {generation} completed in {gen_time:.1f}s "
                  f"({gen_time/60:.1f} min)")

    finally:
        # Ensure workers are cleaned up even if training is interrupted.
        vec_env.close()

    print(f"\n{'='*60}")
    print("Self-play training complete!")
    print(f"Final model: {latest_path}")
    print(f"Pool size: {pool.size()}")
    print(f"Best ELO: {pool.get_best_elo():.0f}")
    print(f"{'='*60}")


def train_vs_random_baseline(
    cards_path: str = CARDS_PATH,
    total_timesteps: int = 2_000_000,
) -> None:
    """Simple training against a random opponent (no self-play pool).

    Useful for initial development, debugging, and establishing a baseline.

    Args:
        cards_path: Path to cards_data.xlsx.
        total_timesteps: Total environment steps to train for.
    """
    print("=" * 60)
    print("Splendor Training vs Random Baseline")
    print("=" * 60)

    # Single process env for simplicity
    env = SelfPlayEnv(
        cards_path=cards_path,
        opponent_model_path=None,  # Random opponent
        agent_player_idx=0,
    )

    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    model = create_agent_model(
        vec_env,
        tensorboard_log=LOG_DIR,
    )

    print(f"Training for {total_timesteps:,} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="vs_random",
    )

    # Save
    baseline_path = os.path.join(CHECKPOINT_DIR, "baseline_vs_random.zip")
    model.save(baseline_path)
    print(f"Saved baseline model: {baseline_path}")

    vec_env.close()
