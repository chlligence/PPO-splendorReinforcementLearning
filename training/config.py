"""Hyperparameters and configuration for Splendor RL training.

All values are tuned for RTX 5070 Ti (16GB VRAM), 24-core CPU, 32GB RAM.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CARDS_PATH = os.path.join(BASE_DIR, "cards_data.xlsx")

if not os.path.exists(CARDS_PATH):
    CARDS_PATH = os.path.join(DATA_DIR, "cards_data.xlsx")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---- PPO Hyperparameters ----
PPO_CONFIG = {
    "n_steps": 2048,            # Steps per environment per rollout
    "batch_size": 512,          # Minibatch size
    "n_epochs": 10,             # PPO epochs per update
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,           # Initial entropy coefficient
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
    "device": "cuda",           # "cuda" or "cpu"
}

# ---- Neural Network Architecture ----
NETWORK_CONFIG = {
    "features_dim": 256,        # Output dim of feature extractor
    "hidden_layers": [512, 512, 512],  # Feature extractor hidden layers
    "pi_layers": [512, 512],    # Policy head hidden layers
    "vf_layers": [512, 512],    # Value head hidden layers
}

# ---- Self-Play Configuration ----
SELFPLAY_CONFIG = {
    "n_envs": 20,                       # Parallel environments (leaves ~4 cores free)
    "generations": 50,                   # Total training generations
    "steps_per_generation": 500_000,     # Environment steps per generation
    "opponent_pool_size": 20,            # Max entries in opponent pool
    "eval_games": 200,                   # Head-to-head games for ELO evaluation
    "eval_interval_generations": 2,     # Evaluate every N generations
    "save_interval_generations": 1,      # Save checkpoint every N generations
    "ent_start": 0.05,                   # Starting entropy coefficient (gen 0)
    "ent_end": 0.005,                    # Final entropy coefficient
    "ent_anneal_generations": 10,        # Generations to anneal entropy
    "random_opponent_prob": 0.10,        # Probability of random opponent (diversity)
    "latest_opponent_prob": 0.50,        # Probability of latest opponent
}

# ---- Environment Configuration ----
ENV_CONFIG = {
    "render_mode": None,                # Disable rendering during training
    "max_turns": 200,                   # Safety cap (Splendor games rarely exceed ~120)
}

# ---- Logging ----
LOG_CONFIG = {
    "tensorboard_log": LOG_DIR,
    "verbose": 1,                       # SB3 verbosity level
    "log_interval": 100,               # Steps between SB3 log outputs
}
