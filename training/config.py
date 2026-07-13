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
if not os.path.exists(CARDS_PATH):
    CARDS_PATH = os.path.join(BASE_DIR, "game-data", "cards_data.xlsx")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---- PPO Hyperparameters ----
PPO_CONFIG = {
    "n_steps": 1024,            # Steps per environment per rollout (halved for 2× update frequency)
    "batch_size": 256,          # Minibatch size (reduced proportionally with n_steps)
    "n_epochs": 4,              # PPO epochs per update (reduced from 5 — less aggressive reuse)
    "learning_rate": 5e-5,      # (halved from 1e-4 — reduces clip_fraction and approx_kl)
    "gamma": 0.99,
    "gae_lambda": 0.92,         # (reduced from 0.95 — lower advantage variance)
    "clip_range": 0.15,         # (reduced from 0.2 — tighter trust region)
    "ent_coef": 0.01,           # Initial entropy coefficient (overridden by anneal schedule)
    "vf_coef": 0.75,            # (increased from 0.5 — stronger value function learning)
    "max_grad_norm": 0.5,       # (reduced from 1.0 — tighter gradient clipping)
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
    "n_envs": 22,                       # Parallel environments (~1 per physical core;
                                         # 24-core Arrow Lake, no HT, sweet spot 22–24)
    "generations": 100,                  # Total training generations (50M steps total)
    "steps_per_generation": 500_000,     # Environment steps per generation
    "opponent_pool_size": 30,            # Max entries in opponent pool (deeper history for 100 gens)
    "eval_games": 100,                   # Head-to-head games for ELO evaluation (reduced from 200)
    "eval_interval_generations": 10,     # Full eval every 10 gens (≈10 full evals over 100 gens)
    "cheap_eval_games": 40,              # Match-pairs for the cheap per-generation win-rate/ELO check
                                         # (40 pairs = 80 games → σ ≈ ±5.6%, down from ±9% at 15)
    "cheap_elo_k": 16.0,                 # Smaller K-factor for the low-sample cheap ELO update
    "save_interval_generations": 1,      # Save checkpoint every N generations
    "ent_start": 0.08,                   # Starting entropy coefficient (increased from 0.05)
    "ent_end": 0.01,                     # Final entropy coefficient (increased from 0.005)
    "ent_anneal_generations": 60,        # Anneal entropy over 60% of training (extended for 100 gens)
    "random_opponent_prob": 0.10,        # Probability of uniform pool opponent (diversity)
    "random_action_prob": 0.05,          # Probability of truly random-action opponent
    "latest_opponent_prob": 0.50,        # Probability of latest opponent
    "elo_temperature": 50.0,             # Temperature for ELO-softmax (ELO-scale units)
}

# ---- Environment Configuration ----
ENV_CONFIG = {
    "render_mode": None,                # Disable rendering during training
    "max_turns": 200,                   # Safety cap (Splendor games rarely exceed ~120)
    "shaping_gamma": 0.99,              # PBRS γ — MUST equal PPO_CONFIG["gamma"]
}

# ---- Logging ----
LOG_CONFIG = {
    "tensorboard_log": LOG_DIR,
    "verbose": 1,                       # SB3 verbosity level
    "log_interval": 100,               # Steps between SB3 log outputs
}
