"""Reward shaping for Splendor RL.

Terminal reward (+1/-1 for win/loss) dominates all shaping rewards.
Auxiliary rewards are kept conservative to avoid local optima.
Specialized for 2-player, No-Noble, infrastructure/deep-digging and
dynamic board-target alignment meta.

Revision note (post gen-49 review): the previous version over-rewarded
raw "infrastructure" accumulation (bonus count / same-color stacking)
independent of whether it ever converted into a high-point purchase, and
had no signal at all for denying the opponent a card they were building
toward. Both were dense and risk-free every turn, so the agent settled
into a stable "hoard tier-1 bonuses, never commit to a big card, never
block" equilibrium against itself in self-play. This revision:
  - cuts the weight of bonus/concentration/alignment accumulation terms,
  - adds a dense "progress toward the best card on the board" term so the
    multi-turn gem-gathering grind for an expensive card is no longer a
    reward-free gap compared to buying cheap cards every turn,
  - adds an explicit denial bonus for reserving a card the opponent was
    actually close to affording,
  - removes the flat reward for merely holding reserved cards (that was
    rewarding hoarding, not usefulness).
"""

import numpy as np

from .constants import MAX_TOTAL_TOKENS, WINNING_POINTS
from .game_state import GameState


def compute_reward(
    prev_state: GameState,
    state: GameState,
    action: int,
    player_idx: int,
) -> float:
    """Compute the shaped reward for the player who just acted.

    Args:
        prev_state: Game state before the action.
        state: Game state after the action.
        action: The action index (0-50) that was taken.
        player_idx: The player who acted (0 or 1).

    Returns:
        Scalar reward value.
    """
    # ---- 1. Terminal reward (dominates all shaping, strictly zero-sum) ----
    if state.game_over:
        if state.winner == player_idx:
            return 1.0    # Win
        elif state.winner == (1 - player_idx):
            return -1.0   # Loss
        else:
            return 0.0    # Draw

    p = state.players[player_idx]
    p_prev = prev_state.players[player_idx]
    opp = state.players[1 - player_idx]
    opp_prev = prev_state.players[1 - player_idx]

    # ---- 2. Dynamic Board Demand Analysis (高分牌主要颜色需求分析) ----
    # 扫描动作执行前版面（prev_state）上 Tier 2 和 Tier 3 的高分卡，计算各颜色的稀缺度/需求度
    demand_weights = np.zeros(5, dtype=np.float32)
    for level in [2, 3]:
        face_up_list = prev_state.face_up.get(level, [])
        for card in face_up_list:
            if card is not None and card.points > 0:
                for c in range(5):
                    # 核心逻辑：将卡牌的各颜色成本乘以其自身的分数，权重向高分核心卡严重倾斜
                    demand_weights[c] += card.cost[c] * card.points

    # 归一化需求向量，使其转化为购买导向的分布概率（和为1.0），防止奖励数值爆炸
    total_demand = np.sum(demand_weights)
    if total_demand > 0:
        demand_weights = demand_weights / total_demand

    # ---- 3. Infrastructure & Strategic Potential Function (综合资产势能函数) ----
    def get_strategic_potential(game_state, player, bonuses_array):
        """评估当前玩家资产的综合战略价值"""
        # 基础分权重 (绝对核心目标)
        pt_score = player.points * 1.0

        # 永久宝石基建权重（下调：0.25→0.08）。基建本身只是通往高分牌的手段，
        # 之前的权重让"攒 bonus"本身就近似有利可图，导致智能体满足于囤基础牌
        # 而不去承受多回合无奖励的攒钱期去换一张高分牌。
        base_bonus_count = sum(bonuses_array)
        bonus_score = base_bonus_count * 0.08

        # 【特性1】同色深挖奖励 (Vertical Deepening)，同样下调 (0.08→0.03)，
        # 原因同上：这是手段性奖励，不应和拿分本身量级相当。
        max_single_color = int(np.max(bonuses_array)) if len(bonuses_array) > 0 else 0
        concentration_bonus = 0.03 * max_single_color if max_single_color >= 3 else 0.0

        # 【特性2】看板高分契合度奖励 (Dynamic Board-Target Alignment)，下调 (0.15→0.08)。
        # 这是"宏观"的颜色专精度信号；真正驱动"去冲某张具体高分牌"的是下面新增的
        # target_score（微观/具体目标进度），两者互补，所以各自权重都调低。
        alignment_score = np.sum(bonuses_array * demand_weights) * 0.08

        # 【特性3，新增】具体目标卡进度奖励 (Target Card Affordability Progress)
        # 找到当前版面上分值最高的 Tier2/3 卡，衡量玩家离买得起它还差多远（0~1，
        # 综合永久加成+持有代币+黄金万能币）。这把"攒钱换大卡"的整个过程从
        # "中途零奖励的空窗期"变成"每回合都有密集正反馈的过程"，从而在奖励密度上
        # 能和"每回合买一张便宜基础牌"的策略正面竞争，而不是天然处于劣势。
        target_progress = _best_target_progress(game_state, player)
        target_score = target_progress * 0.35

        # 普通代币与黄金代币价值 (前5位是普通宝石，第6位是黄金)
        token_score = sum(player.tokens[:5]) * 0.02
        gold_score = player.tokens[5] * 0.025  # 黄金作为万能币，具备更高的弹性和战略防御卡位价值

        # 预留手牌：不再对"持有≤2张预留卡"给予平白奖励（那是在奖励囤卡本身，
        # 而不是奖励囤卡的用途）。只保留对囤满3张不买的轻度节奏惩罚。
        # 真正有价值的预留（卡住对手）由下面第5节的显式"卡位奖励"负责。
        reserve_count = len(player.reserved)
        reserve_score = -0.015 if reserve_count >= 3 else 0.0

        return (
            pt_score + bonus_score + concentration_bonus + alignment_score
            + target_score + token_score + gold_score + reserve_score
        )

    # 计算相对势能差值 (保持 Self-Play 零和趋势，防止互相刷分)
    prev_margin = (
        get_strategic_potential(prev_state, p_prev, p_prev.bonuses)
        - get_strategic_potential(prev_state, opp_prev, opp_prev.bonuses)
    )
    curr_margin = (
        get_strategic_potential(state, p, p.bonuses)
        - get_strategic_potential(state, opp, opp.bonuses)
    )

    # 密集塑造奖励缩放系数 (0.05: 配合更低的学习率5e-5，保持更新稳定性)
    reward = 0.05 * (curr_margin - prev_margin)

    # ---- 4. Behavioral Penalties (行为惩罚：修正原版Bug) ----
    # 根据 rules.py, action >= 30 为买卡(公用版面或预留手牌)
    is_buy = (action >= 30)
    total_tokens_prev = int(np.sum(p_prev.tokens))

    # 惩罚：如果你在拿硬币（非买卡），且拿之前就已经满手牌（>=10），会导致被迫弃牌，属于严重的节奏浪费
    if not is_buy and total_tokens_prev >= MAX_TOTAL_TOKENS:
        reward -= 0.04

    # ---- 5. Exact Bonus Match (战术微观效率引导) ----
    if is_buy:
        card = _get_purchased_card(p, p_prev)
        if card is not None:
            # 检查上一次的永久资产是否就已经能完全白嫖这张卡
            exact = all(
                card.cost[c] <= int(p_prev.bonuses[c])
                for c in range(5)
            )
            if exact:
                reward += 0.02
    # 这段if的作用是在玩家购买卡牌时，检查他们是否能够完全使用之前积累的永久资产（即他们的bonus）来支付这张卡的成本。
    # 如果是这样，这意味着玩家在购买这张卡时没有使用任何额外的代币，而是完全依赖于他们之前的投资。
    # 这种行为被认为是高效的策略，因此给予了一个小的奖励（+0.02），以鼓励玩家在游戏中追求这种高效的购买方式。

    # ---- 6. Denial Bonus (新增：卡位/否决对手奖励) ----
    # 之前的版本完全没有对"预留对手正在冲的高分牌"给予任何信号，导致这个战术
    # 在自我对弈里从未被真正验证过有用，也就学不出来。这里只在"预留明牌"动作
    # (action 15-26) 时触发：用对手在此动作发生前的状态评估他离买这张卡有多近，
    # 越接近、这张卡分值越高，卡位奖励越大；盲拿牌堆（27-29）看不到牌面，不给奖励。
    if 15 <= action <= 26:
        pos = action - 15
        level = (pos // 4) + 1
        slot = pos % 4
        face_up_list = prev_state.face_up.get(level, [])
        reserved_card = face_up_list[slot] if slot < len(face_up_list) else None
        if reserved_card is not None and reserved_card.points > 0:
            opp_progress = _affordability_progress(opp_prev, reserved_card)
            reward += 0.06 * (reserved_card.points / 5.0) * opp_progress

    return reward


def _get_purchased_card(p_curr, p_prev):
    """Get the card that was just purchased (if any).

    Compares the purchased lists to find the newly added card.
    """
    if len(p_curr.purchased) > len(p_prev.purchased):
        return p_curr.purchased[-1]
    return None


def _affordability_progress(player, card) -> float:
    """How close `player` is to being able to afford `card` right now.

    Returns a value in [0, 1]: 0 means the shortfall equals the full cost
    (no relevant tokens/bonuses at all), 1 means they could buy it this
    instant. Permanent bonuses reduce the effective cost first, held gems
    cover what's left, and gold (wildcard) covers any remaining shortfall.
    """
    cost = np.asarray(card.cost, dtype=np.float32)
    effective_cost = np.maximum(cost - player.bonuses.astype(np.float32), 0.0)
    total_needed = float(effective_cost.sum())
    if total_needed <= 0:
        return 1.0  # Bonuses alone already cover it.

    shortfall = np.maximum(effective_cost - player.tokens[:5].astype(np.float32), 0.0)
    total_shortfall = float(shortfall.sum())
    gold_covered = min(float(player.tokens[5]), total_shortfall)
    remaining = total_shortfall - gold_covered
    return 1.0 - (remaining / total_needed)


def _best_target_progress(game_state, player) -> float:
    """Progress toward the single most valuable card worth chasing right now.

    Only Tier 2/3 face-up cards with points > 0 are considered (Tier 1 /
    0-point cards are "infrastructure", not a prize). Among the highest-point
    candidates, ties are broken toward whichever this player is closer to
    affording, so the signal doesn't get stuck pointing at an unreachable
    card while an equally valuable one sits next to it.
    """
    best_points = -1
    best_progress = 0.0
    for level in (2, 3):
        for card in game_state.face_up.get(level, []):
            if card is None or card.points <= 0:
                continue
            progress = _affordability_progress(player, card)
            if card.points > best_points or (
                card.points == best_points and progress > best_progress
            ):
                best_points = card.points
                best_progress = progress
    return best_progress if best_points >= 0 else 0.0
