#!/usr/bin/env python3
"""Verify card data parsing from cards_data.xlsx.

Usage:
    python scripts/verify_cards.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splendor.card import load_cards, get_card_summary


def main():
    xlsx_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "cards_data.xlsx"
    )

    if not os.path.exists(xlsx_path):
        # Try the original file name
        xlsx_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cards_data.xlsx"
        )

    print(f"Loading cards from: {xlsx_path}")
    cards_by_level = load_cards(xlsx_path)

    # Print detailed summary
    print(get_card_summary(cards_by_level))

    # Quick integrity checks
    print("\n--- Integrity Checks ---")
    for level in [1, 2, 3]:
        cards = cards_by_level[level]
        print(f"Level {level}: {len(cards)} cards [OK]")

        # Check all bonuses are valid
        for c in cards:
            assert 0 <= c.bonus <= 4, f"Invalid bonus: {c}"
            assert 0 <= c.points <= 5, f"Invalid points: {c}"
            assert len(c.cost) == 5, f"Invalid cost: {c}"
            for cost_val in c.cost:
                assert cost_val >= 0, f"Negative cost: {c}"

    print("All integrity checks passed! [OK]")

    # Sample a few cards
    print("\n--- Sample Cards ---")
    for level in [1, 2, 3]:
        cards = cards_by_level[level]
        print(f"\nLevel {level} (first 3):")
        for card in cards[:3]:
            print(f"  ID:{card.card_id:02d} Bonus:{card.bonus.name:6s} "
                  f"Pts:{card.points} Cost:{card.cost}")

    total = sum(len(v) for v in cards_by_level.values())
    print(f"\nTotal cards loaded: {total}")


if __name__ == "__main__":
    main()
