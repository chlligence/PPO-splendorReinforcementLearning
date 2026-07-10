"""Flask web server for playing Splendor against a trained MaskablePPO agent.

Single in-memory game session (local, single-user tool). Reuses the existing
`splendor` game engine and the `training.feature_extractor` module needed to
unpickle the SB3 checkpoint.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, request, send_from_directory
import numpy as np
from sb3_contrib import MaskablePPO

from splendor.card import load_cards, Card
from splendor.env import SplendorEnv
from splendor.action_mask import get_action_mask, get_action_description, N_ACTIONS

CARDS_PATH = os.path.join(PROJECT_ROOT, "cards_data.xlsx")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "agent_latest.zip")
GEM_NAMES = ["black", "white", "red", "blue", "green", "gold"]

app = Flask(__name__, static_folder="static", static_url_path="")

CARDS_BY_LEVEL = load_cards(CARDS_PATH)
MODEL = MaskablePPO.load(CHECKPOINT_PATH, device="cpu")

# ---- Single global game session ----
session = {
    "env": None,
    "human_idx": 0,
    "ai_idx": 1,
    "last_ai_actions": [],
    "obs": None,
    "mask": None,
}


def card_to_json(card: Card):
    if card is None:
        return None
    return {
        "card_id": card.card_id,
        "level": card.level,
        "bonus": GEM_NAMES[int(card.bonus)],
        "points": card.points,
        "cost": {GEM_NAMES[i]: int(card.cost[i]) for i in range(5)},
    }


def classify_action(action: int):
    """Return structured info about an action index so the frontend can wire
    it up to board elements without parsing description strings."""
    if 0 <= action <= 9:
        from splendor.constants import COMBO_3_DIFFERENT
        return {"category": "take3", "colors": [GEM_NAMES[c] for c in COMBO_3_DIFFERENT[action]]}
    if 10 <= action <= 14:
        return {"category": "take2", "colors": [GEM_NAMES[action - 10]]}
    if 15 <= action <= 26:
        pos = action - 15
        return {"category": "reserve_face", "level": (pos // 4) + 1, "slot": pos % 4}
    if 27 <= action <= 29:
        return {"category": "reserve_deck", "level": action - 27 + 1}
    if 30 <= action <= 41:
        pos = action - 30
        return {"category": "buy_face", "level": (pos // 4) + 1, "slot": pos % 4}
    if 42 <= action <= 44:
        return {"category": "buy_reserved", "slot": action - 42}
    if 45 <= action <= 49:
        return {"category": "take1", "colors": [GEM_NAMES[action - 45]]}
    return {"category": "pass"}


def legal_actions_json(env: SplendorEnv):
    mask = env.action_masks()
    result = []
    for i in range(N_ACTIONS):
        if mask[i]:
            entry = {"action": i, "description": get_action_description(i)}
            entry.update(classify_action(i))
            result.append(entry)
    return result


def state_to_json():
    env: SplendorEnv = session["env"]
    state = env.state
    human_idx = session["human_idx"]
    ai_idx = session["ai_idx"]

    face_up = {}
    for level in [1, 2, 3]:
        face_up[str(level)] = [card_to_json(c) for c in state.face_up.get(level, [])]

    deck_counts = {str(level): len(state.decks.get(level, [])) for level in [1, 2, 3]}

    gems_available = {GEM_NAMES[i]: int(state.gems_available[i]) for i in range(6)}

    def player_json(idx, reveal_reserved):
        p = state.players[idx]
        data = {
            "tokens": {GEM_NAMES[i]: int(p.tokens[i]) for i in range(6)},
            "bonuses": {GEM_NAMES[i]: int(p.bonuses[i]) for i in range(5)},
            "points": p.points,
            "card_count": p.card_count,
            "purchased": [card_to_json(c) for c in p.purchased],
            "reserved_count": len(p.reserved),
        }
        if reveal_reserved:
            data["reserved"] = [card_to_json(c) for c in p.reserved]
        else:
            data["reserved"] = None
        return data

    players = {
        "human": player_json(human_idx, reveal_reserved=True),
        "ai": player_json(ai_idx, reveal_reserved=False),
    }

    is_game_over = state.game_over
    winner = None
    if is_game_over:
        if state.winner is None:
            winner = "draw"
        else:
            winner = "human" if state.winner == human_idx else "ai"

    is_human_turn = (not is_game_over) and state.current_player == human_idx

    return {
        "gems_available": gems_available,
        "face_up": face_up,
        "deck_counts": deck_counts,
        "players": players,
        "turn_number": state.turn_number,
        "final_round_flag": state.final_round_flag,
        "game_over": is_game_over,
        "winner": winner,
        "is_human_turn": is_human_turn,
        "legal_actions": legal_actions_json(env) if is_human_turn else [],
        "last_ai_actions": session["last_ai_actions"],
    }


def run_ai_turns():
    """Let the model play consecutive turns until it's the human's turn or the game ends."""
    env: SplendorEnv = session["env"]
    ai_idx = session["ai_idx"]
    session["last_ai_actions"] = []

    while (not env.state.game_over) and env.state.current_player == ai_idx:
        mask = env.action_masks()
        action, _ = MODEL.predict(session["obs"], action_masks=mask, deterministic=True)
        action = int(action)
        session["last_ai_actions"].append(get_action_description(action))
        obs, reward, terminated, truncated, info = env.step(action)
        session["obs"] = obs
        session["mask"] = info["action_mask"]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    body = request.get_json(silent=True) or {}
    human_first = bool(body.get("human_first", True))

    session["human_idx"] = 0 if human_first else 1
    session["ai_idx"] = 1 if human_first else 0

    env = SplendorEnv(cards_by_level=CARDS_BY_LEVEL, starting_player=0)
    obs, info = env.reset()
    session["env"] = env
    session["obs"] = obs
    session["mask"] = info["action_mask"]
    session["last_ai_actions"] = []

    run_ai_turns()
    return jsonify(state_to_json())


@app.route("/api/state", methods=["GET"])
def get_state():
    if session["env"] is None:
        return jsonify({"error": "no active game"}), 400
    return jsonify(state_to_json())


@app.route("/api/action", methods=["POST"])
def take_action():
    env: SplendorEnv = session["env"]
    if env is None:
        return jsonify({"error": "no active game"}), 400
    if env.state.game_over:
        return jsonify({"error": "game is over"}), 400
    if env.state.current_player != session["human_idx"]:
        return jsonify({"error": "not human's turn"}), 400

    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action is None or not isinstance(action, int):
        return jsonify({"error": "missing/invalid 'action'"}), 400

    mask = env.action_masks()
    if action < 0 or action >= N_ACTIONS or not mask[action]:
        return jsonify({"error": f"illegal action {action}"}), 400

    obs, reward, terminated, truncated, info = env.step(action)
    session["obs"] = obs
    session["mask"] = info["action_mask"]

    run_ai_turns()
    return jsonify(state_to_json())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
