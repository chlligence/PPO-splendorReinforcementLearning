"""Self-Play training loop for Splendor RL.

Orchestrates iterative training: each generation trains the agent against
a sampled opponent from the pool, then evaluates and adds the new policy
to the pool.

Uses Stable-Baselines3 MaskablePPO with SubprocVecEnv for parallelism.
"""

import os
import time
from typing import Optional
import numpy as np

from splendor.card import load_cards

from .config import (
    PPO_CONFIG, NETWORK_CONFIG, SELFPLAY_CONFIG,
    CHECKPOINT_DIR, LOG_DIR, CARDS_PATH,
)
from .feature_extractor import SplendorFeatureExtractor
from .opponent_pool import OpponentPool, PoolEntry
from .self_play_env import make_env_fn, SelfPlayEnv
from .evaluate import evaluate_head_to_head, estimate_elo


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


def run_self_play(
    cards_path: str = CARDS_PATH,
    resume_from: Optional[str] = None,
) -> None:
    """Run the full self-play training pipeline.

    Args:
        cards_path: Path to cards_data.xlsx.
        resume_from: Path to a checkpoint to resume from (optional).
    """
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
        random_prob=SELFPLAY_CONFIG["random_opponent_prob"],
    )
    pool_index_path = os.path.join(CHECKPOINT_DIR, "pool_index.json")
    if os.path.exists(pool_index_path):
        pool.load_index(pool_index_path, CHECKPOINT_DIR)

    rng = np.random.default_rng()

    # Determine starting generation
    start_gen = 0
    agent_model = None

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

        # 2. Create vectorized environments
        print(f"Creating {SELFPLAY_CONFIG['n_envs']} parallel environments...")
        env_fns = [
            make_env_fn(
                cards_path=cards_path,
                opponent_model_path=opponent_path,
                agent_player_idx=0,
                rank=generation * 1000 + i,
            )
            for i in range(SELFPLAY_CONFIG['n_envs'])
        ]

        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv(env_fns)

        # 3. Create or update agent model
        ent_coef = compute_entropy_coef(generation)
        print(f"Entropy coefficient: {ent_coef:.4f}")

        if agent_model is None:
            agent_model = create_agent_model(
                vec_env,
                features_dim=NETWORK_CONFIG["features_dim"],
                learning_rate=PPO_CONFIG["learning_rate"],
                ent_coef=ent_coef,
                device=PPO_CONFIG["device"],
                tensorboard_log=LOG_DIR,
            )
        else:
            agent_model.set_env(vec_env)
            agent_model.ent_coef = ent_coef
            # Update learning rate (can be scheduled)
            agent_model.learning_rate = PPO_CONFIG["learning_rate"]

        # 4. Train
        print(f"Training for {SELFPLAY_CONFIG['steps_per_generation']:,} steps...")
        agent_model.learn(
            total_timesteps=SELFPLAY_CONFIG["steps_per_generation"],
            reset_num_timesteps=False,
            tb_log_name=f"gen_{generation}",
        )

        # 5. Save checkpoint
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f"agent_gen_{generation}.zip"
        )
        latest_path = os.path.join(CHECKPOINT_DIR, "agent_latest.zip")
        agent_model.save(checkpoint_path)
        agent_model.save(latest_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # 6. Evaluate periodically
        eval_interval = SELFPLAY_CONFIG["eval_interval_generations"]
        if generation % eval_interval == 0 and pool.size() > 0:
            print(f"Evaluating vs pool (ELO estimation)...")
            elo = estimate_elo(agent_model, pool, cards_path,
                               num_games=SELFPLAY_CONFIG["eval_games"])
            print(f"Estimated ELO: {elo:.0f}")

            # Quick eval vs latest for win_rate_vs_prev
            from sb3_contrib import MaskablePPO
            latest_opp = MaskablePPO.load(pool.entries[-1].path)
            results = evaluate_head_to_head(
                agent_model, latest_opp, cards_path, num_games=50,
            )
            win_rate_vs_prev = results["a_win_rate"]
            print(f"Win rate vs latest opponent: {win_rate_vs_prev:.2%}")
        else:
            # Use default values for generations without full eval
            elo = pool.get_best_elo() + 10 if pool.size() > 0 else 1200.0
            win_rate_vs_prev = 0.5

        # 7. Add to pool
        pool.add(PoolEntry(
            path=checkpoint_path,
            generation=generation,
            elo=elo,
            win_rate_vs_prev=win_rate_vs_prev,
        ))
        pool.save_index(pool_index_path)

        # Clean up envs
        vec_env.close()

        gen_time = time.time() - gen_start_time
        print(f"Generation {generation} completed in {gen_time:.1f}s "
              f"({gen_time/60:.1f} min)")

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
