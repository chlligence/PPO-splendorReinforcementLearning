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
from splendor.constants import COMBO_3_DIFFERENT, NUM_COLORS, MAX_TOTAL_TOKENS

CARDS_PATH = os.path.join(PROJECT_ROOT, "game-data/cards_data.xlsx")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "agent_latest.zip")
GEM_NAMES = ["black", "white", "red", "blue", "green", "gold"]
GEM_NAMES_CN = ["黑曜", "钻石", "红宝", "蓝宝", "祖母绿", "黄金"]

app = Flask(__name__, static_folder="static", static_url_path="")

CARDS_BY_LEVEL = load_cards(CARDS_PATH)
MODEL = MaskablePPO.load(CHECKPOINT_PATH, device="cpu")

# ---- Single global game session ----
session = {
    "env": None,
    "human_idx": 0,
    "ai_idx": 1,
    "ai_action_log": [],       # accumulated structured AI action entries
    "obs": None,
    "mask": None,
    "pending_return": None,    # {action, excess_count, gems_taken} for gem return flow
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


def get_card_for_action(env: SplendorEnv, action: int) -> Card | None:
    """Return the card that an action will buy or reserve (before execution).

    Returns None for non-card actions (take gems, pass).
    """
    state = env.state
    human_idx = session["human_idx"]
    ai_idx = session["ai_idx"]
    p = state.players[ai_idx]

    if 15 <= action <= 26:  # reserve face-up
        pos = action - 15
        level = (pos // 4) + 1
        slot = pos % 4
        face_up_list = state.face_up.get(level, [])
        if slot < len(face_up_list):
            return face_up_list[slot]
    elif 27 <= action <= 29:  # reserve deck
        level = action - 27 + 1
        deck = state.decks.get(level, [])
        if deck:
            return deck[-1]  # top of deck
    elif 30 <= action <= 41:  # buy face-up
        pos = action - 30
        level = (pos // 4) + 1
        slot = pos % 4
        face_up_list = state.face_up.get(level, [])
        if slot < len(face_up_list):
            return face_up_list[slot]
    elif 42 <= action <= 44:  # buy reserved
        slot = action - 42
        if slot < len(p.reserved):
            return p.reserved[slot]

    return None


def legal_actions_json(env: SplendorEnv):
    mask = env.action_masks()
    result = []
    for i in range(N_ACTIONS):
        if mask[i]:
            entry = {"action": i, "description": get_action_description(i)}
            entry.update(classify_action(i))
            result.append(entry)
    return result


def _compute_tokens_after_take(env: SplendorEnv, action: int):
    """Compute what the player's tokens would look like after a take action.

    Returns (tokens_after, gems_taken_dict) or (None, None) if not a take action.
    gems_taken_dict maps gem_name -> count taken.
    """
    state = env.state
    human_idx = session["human_idx"]
    current_tokens = state.players[human_idx].tokens.copy()
    gems_taken = {}

    if 0 <= action <= 9:  # take 3 different
        colors = COMBO_3_DIFFERENT[action]
        for c in colors:
            current_tokens[c] += 1
            name = GEM_NAMES[c]
            gems_taken[name] = gems_taken.get(name, 0) + 1
    elif 10 <= action <= 14:  # take 2 same
        c = action - 10
        current_tokens[c] += 2
        gems_taken[GEM_NAMES[c]] = 2
    elif 45 <= action <= 49:  # take 1 gem
        c = action - 45
        current_tokens[c] += 1
        gems_taken[GEM_NAMES[c]] = 1
    else:
        return None, None

    return current_tokens, gems_taken


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

    result = {
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
        "ai_action_log": session["ai_action_log"],
    }

    # Include pending return info if applicable
    if session["pending_return"] is not None:
        pending = session["pending_return"]
        human_p = state.players[human_idx]
        current_tokens = {GEM_NAMES[i]: int(human_p.tokens[i]) for i in range(6)}
        # Compute tokens_after_take
        tokens_after, gems_taken = _compute_tokens_after_take(env, pending["action"])
        tokens_after_dict = None
        if tokens_after is not None:
            tokens_after_dict = {GEM_NAMES[i]: int(tokens_after[i]) for i in range(6)}
        result["needs_return"] = True
        result["pending_action"] = pending["action"]
        result["excess_count"] = pending["excess_count"]
        result["current_tokens"] = current_tokens
        result["tokens_after_take"] = tokens_after_dict
        result["gems_taken"] = gems_taken

    return result


def run_ai_turns():
    """Let the model play consecutive turns until it's the human's turn or the game ends.

    For each AI action, captures structured info (category, card details for
    buy/reserve) and appends it to the accumulated ai_action_log.
    """
    env: SplendorEnv = session["env"]
    ai_idx = session["ai_idx"]

    while (not env.state.game_over) and env.state.current_player == ai_idx:
        mask = env.action_masks()

        # Snapshot card info BEFORE the action (the action will consume the card)
        action, _ = MODEL.predict(session["obs"], action_masks=mask, deterministic=True)
        action = int(action)
        card = get_card_for_action(env, action)
        category = classify_action(action)["category"]
        description = get_action_description(action)

        # Build structured log entry
        entry = {
            "turn_number": int(env.state.turn_number),
            "player": "ai",
            "action": action,
            "category": category,
            "description": description,
        }
        if card is not None and category in ("buy_face", "buy_reserved", "reserve_face", "reserve_deck"):
            entry["card_info"] = card_to_json(card)

        # Execute the action
        obs, reward, terminated, truncated, info = env.step(action)
        session["obs"] = obs
        session["mask"] = info["action_mask"]

        # Append to accumulated log
        session["ai_action_log"].append(entry)


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
    session["ai_action_log"] = []
    session["pending_return"] = None

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

    category = classify_action(action)["category"]

    # ---- Overflow check for take actions ----
    if category in ("take3", "take2", "take1"):
        human_idx = session["human_idx"]
        current_total = int(np.sum(env.state.players[human_idx].tokens))
        tokens_after, gems_taken = _compute_tokens_after_take(env, action)

        if tokens_after is not None:
            new_total = int(np.sum(tokens_after))
            if new_total > MAX_TOTAL_TOKENS:
                # Store pending return and let frontend handle gem selection
                excess = new_total - MAX_TOTAL_TOKENS
                session["pending_return"] = {
                    "action": action,
                    "excess_count": excess,
                    "gems_taken": gems_taken,
                }
                return jsonify(state_to_json())

    # ---- Normal action execution (buy, reserve, pass, or take without overflow) ----
    obs, reward, terminated, truncated, info = env.step(action)
    session["obs"] = obs
    session["mask"] = info["action_mask"]
    session["pending_return"] = None

    # Append human action to log (optional — for completeness)
    desc = get_action_description(action)
    session["ai_action_log"].append({
        "turn_number": int(env.state.turn_number),
        "player": "human",
        "action": action,
        "category": category,
        "description": desc,
    })

    run_ai_turns()
    return jsonify(state_to_json())


@app.route("/api/return_gems", methods=["POST"])
def return_gems():
    """Handle gem return after a take action caused token overflow.

    Receives: {action: int, return_colors: [string]}
    The action must match the stored pending_return action.
    """
    env: SplendorEnv = session["env"]
    if env is None:
        return jsonify({"error": "no active game"}), 400
    if env.state.game_over:
        return jsonify({"error": "game is over"}), 400

    pending = session["pending_return"]
    if pending is None:
        return jsonify({"error": "no pending return"}), 400

    body = request.get_json(silent=True) or {}
    action = body.get("action")
    return_colors = body.get("return_colors", [])

    if action is None or not isinstance(action, int):
        return jsonify({"error": "missing/invalid 'action'"}), 400
    if action != pending["action"]:
        return jsonify({"error": "action does not match pending return"}), 400
    if not isinstance(return_colors, list):
        return jsonify({"error": "return_colors must be a list"}), 400
    if len(return_colors) != pending["excess_count"]:
        return jsonify({"error": f"must return exactly {pending['excess_count']} gems, got {len(return_colors)}"}), 400

    human_idx = session["human_idx"]
    state = env.state
    p = state.players[human_idx]

    # Validate return_colors before executing
    # Build the post-take token counts to check if player has enough to return
    tokens_after, _ = _compute_tokens_after_take(env, action)
    if tokens_after is None:
        return jsonify({"error": "invalid take action"}), 400

    color_indices = {name: i for i, name in enumerate(GEM_NAMES)}
    for color_name in return_colors:
        if color_name not in color_indices:
            return jsonify({"error": f"invalid color: {color_name}"}), 400
        idx = color_indices[color_name]
        if tokens_after[idx] <= 0:
            return jsonify({"error": f"cannot return {color_name}: player has none after take"}), 400
        # Deduct from the projected tokens so we validate correctly
        tokens_after[idx] -= 1

    # ---- Execute: add gems from take action ----
    category = classify_action(action)["category"]
    if category == "take3":
        colors = COMBO_3_DIFFERENT[action]
        for c in colors:
            state.gems_available[c] -= 1
            p.tokens[c] += 1
    elif category == "take2":
        c = action - 10
        state.gems_available[c] -= 2
        p.tokens[c] += 2
    elif category == "take1":
        c = action - 45
        state.gems_available[c] -= 1
        p.tokens[c] += 1

    # ---- Remove returned gems ----
    for color_name in return_colors:
        idx = color_indices[color_name]
        p.tokens[idx] -= 1
        state.gems_available[idx] += 1

    # ---- Clear pending return ----
    session["pending_return"] = None

    # ---- Append human action to log ----
    desc = get_action_description(action)
    session["ai_action_log"].append({
        "turn_number": int(state.turn_number),
        "player": "human",
        "action": action,
        "category": category,
        "description": desc,
        "returned_gems": return_colors,
    })

    # ---- Run AI turns ----
    run_ai_turns()
    return jsonify(state_to_json())


@app.route("/api/cancel_return", methods=["POST"])
def cancel_return():
    """Cancel the pending gem return and restore normal play.

    Clears the pending_return state so the player can choose a different action.
    Does NOT consume the player's turn — no action is executed.
    """
    if session["env"] is None:
        return jsonify({"error": "no active game"}), 400
    if session["pending_return"] is None:
        return jsonify({"error": "no pending return"}), 400

    session["pending_return"] = None
    return jsonify(state_to_json())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
