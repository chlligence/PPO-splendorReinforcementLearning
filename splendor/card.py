"""Card dataclass and deck loader for Splendor."""

from dataclasses import dataclass
from typing import Dict, List
import openpyxl

from .constants import Gem, CARDS_PER_LEVEL

# Mapping from spreadsheet bonus strings to Gem enum
_GEM_MAP = {
    "Black": Gem.BLACK,
    "White": Gem.WHITE,
    "Red": Gem.RED,
    "Blue": Gem.BLUE,
    "Green": Gem.GREEN,
}


@dataclass(frozen=True)
class Card:
    """Immutable representation of a Splendor card.

    Attributes:
        card_id: Unique identifier (0-89).
        level: Card tier (1, 2, or 3).
        bonus: Gem color this card produces as a permanent discount.
        points: Victory points (0-5).
        cost: Tuple of (black, white, red, blue, green) gem costs.
    """
    card_id: int
    level: int
    bonus: Gem
    points: int
    cost: tuple  # (int, int, int, int, int) for Black, White, Red, Blue, Green


def load_cards(xlsx_path: str) -> Dict[int, List[Card]]:
    """Load all cards from the Splendor cards_data.xlsx file.

    Args:
        xlsx_path: Path to cards_data.xlsx.

    Returns:
        Dict mapping level (1, 2, 3) to list of Card objects.
        Level 1: 40 cards, Level 2: 30 cards, Level 3: 20 cards.

    Raises:
        ValueError: If card counts don't match expected values.
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active

    cards_by_level: Dict[int, List[Card]] = {1: [], 2: [], 3: []}
    card_id = 0

    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue  # Skip empty rows

        level = int(row[0])
        bonus_str = str(row[1]).strip()
        points = int(row[2])
        # Columns D-H: Black, White, Red, Blue, Green cost values
        black = int(row[3])
        white = int(row[4])
        red = int(row[5])
        blue = int(row[6])
        green = int(row[7])

        card = Card(
            card_id=card_id,
            level=level,
            bonus=_GEM_MAP[bonus_str],
            points=points,
            cost=(black, white, red, blue, green),
        )
        cards_by_level[level].append(card)
        card_id += 1

    # Validate counts
    for lvl, expected in CARDS_PER_LEVEL.items():
        actual = len(cards_by_level[lvl])
        if actual != expected:
            raise ValueError(
                f"Level {lvl}: expected {expected} cards, got {actual}"
            )

    return cards_by_level


def get_card_summary(cards_by_level: Dict[int, List[Card]]) -> str:
    """Generate a human-readable summary of the card database.

    Args:
        cards_by_level: Output of load_cards().

    Returns:
        Multi-line summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Splendor Card Database Summary")
    lines.append("=" * 60)

    total = 0
    for level in [1, 2, 3]:
        cards = cards_by_level[level]
        total += len(cards)
        lines.append(f"\n--- Level {level} ({len(cards)} cards) ---")

        # Point distribution
        pts_dist = {}
        for c in cards:
            pts_dist[c.points] = pts_dist.get(c.points, 0) + 1
        pts_str = ", ".join(f"{p}pts×{n}" for p, n in sorted(pts_dist.items()))
        lines.append(f"  Points: {pts_str}")

        # Bonus distribution
        bonus_dist = {}
        for c in cards:
            bname = c.bonus.name
            bonus_dist[bname] = bonus_dist.get(bname, 0) + 1
        bonus_str = ", ".join(f"{k}×{v}" for k, v in sorted(bonus_dist.items()))
        lines.append(f"  Bonuses: {bonus_str}")

        # Cost range
        total_costs = [sum(c.cost) for c in cards]
        max_single_costs = [max(c.cost) for c in cards]
        lines.append(f"  Total cost range: {min(total_costs)}–{max(total_costs)}")
        lines.append(f"  Max single-color cost: {max(max_single_costs)}")

    lines.append(f"\nTotal cards: {total}")
    lines.append("=" * 60)
    return "\n".join(lines)
