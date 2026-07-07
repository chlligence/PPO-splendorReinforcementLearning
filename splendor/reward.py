# """Reward shaping for Splendor RL.

# Terminal reward (+1/-1 for win/loss) dominates all shaping rewards.
# Auxiliary rewards are kept conservative to avoid local optima.
# """

# import numpy as np

# from .constants import MAX_TOTAL_TOKENS, WINNING_POINTS
# from .game_state import GameState


# def compute_reward(
#     prev_state: GameState,
#     state: GameState,
#     action: int,
#     player_idx: int,
# ) -> float:
#     """Compute the shaped reward for the player who just acted.

#     Args:
#         prev_state: Game state before the action.
#         state: Game state after the action.
#         action: The action index (0-44) that was taken.
#         player_idx: The player who acted (0 or 1).

#     Returns:
#         Scalar reward value.
#     """
#     p = state.players[player_idx]
#     p_prev = prev_state.players[player_idx]

#     # ---- Terminal reward (dominates all shaping) ----
#     if state.game_over:
#         if state.winner == player_idx:
#             return 1.0    # Win
#         elif state.winner == (1 - player_idx):
#             return -1.0   # Loss
#         else:
#             return 0.0    # Draw

#     reward = 0.0

#     # ---- Buy card: small positive proportional to points gained ----
#     if action >= 30:  # Any buy action (face-up or reserved)
#         points_gained = p.points - p_prev.points
#         reward += 0.05 * points_gained

#     # ---- First to trigger final round: bonus for reaching winning position ----
#     if state.final_round_flag and not prev_state.final_round_flag:
#         if state.final_round_player == player_idx:
#             reward += 0.3

#     # ---- Token efficiency penalty: discourage hoarding ----
#     total_tokens = int(np.sum(p.tokens))
#     if total_tokens > 8:
#         reward -= 0.01 * (total_tokens - 8)

#     # ---- Exact bonus match: reward for buying a card entirely with bonuses ----
#     if action >= 30:
#         card = _get_purchased_card(p, p_prev)
#         if card is not None:
#             # Check if bonuses alone cover the full cost
#             exact = all(
#                 card.cost[c] <= int(p_prev.bonuses[c])
#                 for c in range(5)
#             )
#             if exact:
#                 reward += 0.02

#     # ---- Penalty for buying while at max tokens ----
#     if total_tokens >= MAX_TOTAL_TOKENS and action >= 30:
#         reward -= 0.03

#     return reward

"""Reward shaping for Splendor RL.

Terminal reward (+1/-1 for win/loss) dominates all shaping rewards.
Auxiliary rewards are kept conservative to avoid local optima.
Specialized for 2-player, No-Noble, infrastructure/deep-digging and 
dynamic board-target alignment meta.
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
    def get_strategic_potential(player, bonuses_array):
        """评估当前玩家资产的综合战略价值"""
        # 基础分权重 (绝对核心目标)
        pt_score = player.points * 1.0
        
        # 永久宝石基建权重 (无贵族局中，基建是冲高分唯一的跳板，给予高基础分 0.25)
        base_bonus_count = sum(bonuses_array)
        bonus_score = base_bonus_count * 0.25
        
        # 【特性1】同色深挖奖励 (Vertical Deepening)
        # 两人局单色硬币上限极低(4枚)，某一色bonus达到3个以上时，战略威慑力和购买力发生质变
        max_single_color = int(np.max(bonuses_array)) if len(bonuses_array) > 0 else 0
        concentration_bonus = 0.08 * max_single_color if max_single_color >= 3 else 0.0
        
        # 【特性2】看板高分契合度奖励 (Dynamic Board-Target Alignment)
        # 鼓励智能体购买的每一张基建卡，都完美对齐版面上当前高分牌所需要的颜色
        alignment_score = np.sum(bonuses_array * demand_weights) * 0.15
        
        # 普通代币与黄金代币价值 (前5位是普通宝石，第6位是黄金)
        token_score = sum(player.tokens[:5]) * 0.02
        gold_score = player.tokens[5] * 0.025  # 黄金作为万能币，具备更高的弹性和战略防御卡位价值
        
        # 预留手牌限制
        reserve_count = len(player.reserved)
        reserve_score = reserve_count * 0.02 if reserve_count <= 2 else -0.02 # 囤积3张不买算卡手
        
        return pt_score + bonus_score + concentration_bonus + alignment_score + token_score + gold_score + reserve_score

    # 计算相对势能差值 (保持 Self-Play 零和趋势，防止互相刷分)
    prev_margin = get_strategic_potential(p_prev, p_prev.bonuses) - get_strategic_potential(opp_prev, opp_prev.bonuses)
    curr_margin = get_strategic_potential(p, p.bonuses) - get_strategic_potential(opp, opp.bonuses)
    
    # 密集塑造奖励缩放系数
    reward = 0.04 * (curr_margin - prev_margin)

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
    #这段if的作用是在玩家购买卡牌时，检查他们是否能够完全使用之前积累的永久资产（即他们的bonus）来支付这张卡的成本。如果是这样，这意味着玩家在购买这张卡时没有使用任何额外的代币，而是完全依赖于他们之前的投资。这种行为被认为是高效的策略，因此给予了一个小的奖励（+0.02），以鼓励玩家在游戏中追求这种高效的购买方式。

    return reward



def _get_purchased_card(p_curr, p_prev):
    """Get the card that was just purchased (if any).

    Compares the purchased lists to find the newly added card.
    """
    if len(p_curr.purchased) > len(p_prev.purchased):
        return p_curr.purchased[-1]
    return None
