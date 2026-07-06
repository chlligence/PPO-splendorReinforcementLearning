"""Game constants and enumerations for Splendor."""

from enum import IntEnum
import numpy as np


class Gem(IntEnum):
    """Gem colors. GOLD is the wildcard/joker token."""
    BLACK = 0
    WHITE = 1
    RED = 2
    BLUE = 3
    GREEN = 4
    GOLD = 5


# Human-readable names
GEM_NAMES = ["Black", "White", "Red", "Blue", "Green", "Gold"]
GEM_NAMES_CN = ["黑", "白", "红", "蓝", "绿", "黄金"]

# Number of base gem colors (excluding gold)
NUM_COLORS = 5

# 2-player resource limits
MAX_GEMS_PER_COLOR = 4       # Each base color: 4 tokens
MAX_GOLD = 5                 # Gold joker tokens
INITIAL_GEMS = np.array([4, 4, 4, 4, 4, 5], dtype=np.int32)

# Player limits
MAX_RESERVED_CARDS = 3       # Max cards in hand
MAX_TOTAL_TOKENS = 10        # Must discard down to 10 at turn end

# Card deck sizes
CARDS_PER_LEVEL = {1: 40, 2: 30, 3: 20}
FACE_UP_PER_LEVEL = 4        # 4 face-up cards per level

# Victory condition
WINNING_POINTS = 15           # Triggers final round

# All combinations of 3 different colors (C(5,3) = 10)
COMBO_3_DIFFERENT = [
    (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 2, 3), (0, 2, 4), (0, 3, 4),
    (1, 2, 3), (1, 2, 4), (1, 3, 4),
    (2, 3, 4),
]

# Maximum values for normalisation
MAX_BONUS = 15          # Reasonable upper bound for bonuses of one color
MAX_POINTS_NORM = 15.0  # Normalisation cap for points (same as winning threshold)
MAX_COST_NORM = 7.0     # Max single-color cost across all cards is 7
