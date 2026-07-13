"""Reward function for 2-player, no-nobles Splendor PPO training.

架构(混合式,与最终确认一致)
--------------------------------------------------------------------------
reward =  γ·Φ(s') − Φ(s)          # 势函数骨架:points + bonus 对齐度,会望远镜求和
        + build_bonus             # 买到性价比≥0.5 的构筑牌,加和(非势函数)
        + block/restore reward    # 卡人 / 防卡 restore 的 if 分支,加和(非势函数)
        + terminal(±10 / 0)
        + step / pass / hoarding penalties

势函数(相对、零和):
    Φ(s) = Strength(self) − Strength(opp)
    Strength(p) = w_point(n)·p.points  +  w_bonus(n)·bonus_alignment(p)

其中
  * n            = 行动方(AI)的购牌数;w_point 随 n 上升、w_bonus 随 n 下降
                   一次 compute_reward内 n 冻结,Φ(prev) 与 Φ(curr) 用同一组权重,消除买牌造成的权重跳变。
  * demand       = 版面 tier2/3 中性价比≥0.5 的牌,用 cost[c]·points 加权归一化得到
                   的 5 维"颜色需求"。一次调用内只算一次(取自 prev_state),
                   Φ(prev)/Φ(curr) 共用,消除翻牌噪声。
  * bonus_alignment(p) = demand · p.bonuses   (点积,衡量 p 的引擎多契合版面核心牌)

性价比(cost-performance)= points_eff / relative_cost(带 bonus 折扣),
  其中 0 分牌的 points_eff = 0.2。**性价比只用于筛选,不进任何 reward 求和。**

--------------------------------------------------------------------------
需要你核对的接口假设(字段名不同就改这里)
--------------------------------------------------------------------------
  state.face_up[level]        -> list[Card],level ∈ {1,2,3}
  state.players[i].points     -> int
  state.players[i].bonuses    -> np.ndarray[NUM_COLORS]      各色永久折扣数
  state.players[i].tokens     -> np.ndarray[NUM_COLORS+1]    含金币(index=NUM_COLORS)
  state.players[i].purchased  -> list[Card]                  已购(用于 n 与买牌检测)
  state.players[i].reserved   -> list[Card]                  预留
  state.turn_number / game_over / winner
  Card.points, Card.cost[NUM_COLORS], Card 的 bonus 颜色 -> 见 _card_color()

--------------------------------------------------------------------------
需要你调 / 我替你拍板的地方(都集中在下方常量 + 注释)
--------------------------------------------------------------------------
  1. w_point / w_bonus 的起止值与饱和购牌数 N_SAT —— 用的是保守默认,先跑再调。
  2. BLOCK_SIGN —— 卡人奖励符号。你的原话带负号,但卡人应鼓励=正,故默认 +1。
  3. demand 的性价比用"行动方(prev)bonuses"做折扣;好牌判定各用相应玩家 bonuses。
  4. n 取自 prev_state 的行动方购牌数(冻结在整次调用)。
"""

from collections import Counter

import numpy as np

from .card import Card
from .constants import NUM_COLORS
from .game_state import GameState, PlayerState

GOLD_INDEX = NUM_COLORS  # 金币在 tokens 里的下标

# ===========================================================================
# Base (non-shaping) rewards
# ===========================================================================
TERMINAL_WIN   = 10.0
TERMINAL_LOSS  = -10.0
TERMINAL_DRAW  = 0.0

STEP_PENALTY     = -0.005   # 每个非终局动作(效率激励)
PASS_PENALTY     = -0.5    # pass 应为最后手段
HOARDING_PENALTY = -0.30    # 已达/超过 10 枚代币上限时还拿币

# ===========================================================================
# Shaping / weighting configuration
# ===========================================================================
# *** 必须等于 PPO 训练器的 γ,否则 PBRS 的策略不变性不成立。***
#     训练启动时会做一致性检查 (见 self_play_loop.run_self_play)。
#     注意: 运行时 gamma 由 env.py 的 shaping_gamma 参数显式传入——此处的
#     SHAPING_GAMMA 仅作为 compute_reward() 的默认参数值，正常情况下不会被走到。
#     同步修改: PPO_CONFIG["gamma"], ENV_CONFIG["shaping_gamma"], rewardcc.SHAPING_GAMMA。
SHAPING_GAMMA = 0.99

# 终局用 F = −Φ(prev) 收口,让骨架部分严格望远镜求和到 −Φ(s0)≈0。
# 若终局方差过大(大比分领先让 +10 显小),设 False 换回干净 ±10。
APPLY_TERMINAL_SHAPING = True

# --- w_point(n) / w_bonus(n):随行动方购牌数 n 线性变化 ---------------------
# 前者逐渐变高(后期重分数),后者逐渐下降(前期重 bonus)。n 到 N_SAT 饱和。
# NOTE: 2 人局到 15 分大约买 10~18 张,N_SAT=15 大致能走完;若典型购牌数明显不同请改。
N_SAT          = 15
W_POINT_START  = 0.05
W_POINT_END    = 0.15
W_BONUS_START  = 0.15
W_BONUS_END    = 0.03

# --- 性价比 / 好牌 / demand ------------------------------------------------
CP_THRESHOLD        = 0.5    # 性价比阈值:好牌 & demand 入选门槛
ZERO_POINT_CP_POINTS = 0.2   # 0 分牌在性价比里视作 0.2 分

# --- 加和式附加奖励 --------------------------------------------------------
BUILD_BONUS = 0.20   # 每买到一张性价比≥0.5 的构筑牌
BLOCK_WEIGHT = 1.0   # 卡人/restore 奖励 = BLOCK_WEIGHT · 该牌的边际价值估计
BLOCK_SIGN   = 1.0   # 卡人成功→正奖励。若你确实想要负号,改成 -1.0 即可。


# ===========================================================================
# Low-level helpers
# ===========================================================================
def _relative_cost(card: Card, bonuses: np.ndarray) -> float:
    """折扣后有效成本,下限 0.5(防除零)。"""
    total = 0.0
    for c in range(NUM_COLORS):
        deficit = int(card.cost[c]) - int(bonuses[c])
        if deficit > 0:
            total += deficit
    return max(total, 0.5)


def _cost_performance(card: Card, bonuses: np.ndarray) -> float:
    """性价比 = points_eff / relative_cost(带 bonus 折扣)。0 分牌 points_eff=0.2。

    仅用于筛选(demand 入选、好牌判定),绝不进入任何 reward 求和。
    """
    pts = float(card.points) if card.points > 0 else ZERO_POINT_CP_POINTS
    return pts / _relative_cost(card, bonuses)


def _card_color(card: Card) -> int:
    """卡牌提供的 bonus 颜色(0..NUM_COLORS-1),取不到返回 -1。

    Splendor 每张牌恰好给 1 种颜色的永久 bonus。若你的 Card 字段名不同,改这里。
    """
    for attr in ("color", "gem", "bonus_color", "bonus"):
        if hasattr(card, attr):
            try:
                v = int(getattr(card, attr))
                if 0 <= v < NUM_COLORS:
                    return v
            except (TypeError, ValueError):
                pass
    return -1


def _all_face_up(state: GameState) -> list:
    """摊平所有明牌。"""
    cards = []
    for level in (1, 2, 3):
        for card in state.face_up.get(level, []):
            if card is not None:
                cards.append(card)
    return cards


def _weights_for_n(n: int):
    """w_point(n) 上升 / w_bonus(n) 下降,n∈[0, N_SAT] 线性插值后饱和。"""
    frac = min(float(max(n, 0)) / float(N_SAT), 1.0)
    w_point = W_POINT_START + (W_POINT_END - W_POINT_START) * frac
    w_bonus = W_BONUS_START + (W_BONUS_END - W_BONUS_START) * frac
    return w_point, w_bonus


# ===========================================================================
# demand / bonus alignment / potential Φ
# ===========================================================================
def _demand_weights(state: GameState, bonuses: np.ndarray) -> np.ndarray:
    """版面 tier2/3 中性价比≥阈值的牌,用 cost[c]·points_eff 加权,归一化到 sum=1。

    表达"当前版面上有价值的高分牌都需要哪些颜色"。一次 compute_reward 内只算一次
    (来自 prev_state),供 Φ(prev)/Φ(curr) 共用,消除翻牌噪声。
    """
    w = np.zeros(NUM_COLORS, dtype=float)
    for level in (2, 3):
        for card in state.face_up.get(level, []):
            if card is None:
                continue
            if _cost_performance(card, bonuses) >= CP_THRESHOLD:
                pts = float(card.points) if card.points > 0 else ZERO_POINT_CP_POINTS
                for c in range(NUM_COLORS):
                    w[c] += float(card.cost[c]) * pts
    s = w.sum()
    if s > 0.0:
        w /= s
    return w


def _bonus_alignment(player: PlayerState, demand: np.ndarray) -> float:
    """demand · player.bonuses —— 玩家引擎与版面颜色需求的契合度。"""
    b = np.asarray(player.bonuses, dtype=float)[:NUM_COLORS]
    return float(np.dot(demand, b))


def _strength(player: PlayerState, demand: np.ndarray,
              w_point: float, w_bonus: float) -> float:
    """单个玩家局面强度(越高越好)。"""
    return w_point * float(player.points) + w_bonus * _bonus_alignment(player, demand)


def _potential(state: GameState, player_idx: int, demand: np.ndarray,
               w_point: float, w_bonus: float) -> float:
    """相对势函数 Φ(s) = Strength(self) − Strength(opp),从 player_idx 视角。"""
    me = state.players[player_idx]
    opp = state.players[1 - player_idx]
    return (_strength(me, demand, w_point, w_bonus)
            - _strength(opp, demand, w_point, w_bonus))


def _card_marginal_value(card: Card, demand: np.ndarray,
                         w_point: float, w_bonus: float) -> float:
    """买下 card 对任意玩家 Strength 的边际增量(与 Φ 同量纲、玩家无关)。

        ΔStrength = w_point · card.points  +  w_bonus · demand[card_color]

    * 分数项:该玩家 points 增加 card.points。
    * bonus 项:该玩家在 card 颜色上 +1 bonus,bonus_alignment 增加 demand[color]。

    这就是"对手买下这张牌的估计值"——卡人时我们把它从对手手里抹掉,故据此给奖励。
    """
    color = _card_color(card)
    d = float(demand[color]) if 0 <= color < NUM_COLORS else 0.0
    return w_point * float(card.points) + w_bonus * d


# ===========================================================================
# 动作检测(prev→curr 差分)
# ===========================================================================
def _card_key(card: Card):
    """结构键,用于跨状态匹配同一张牌。"""
    return (int(card.points), tuple(int(x) for x in card.cost), _card_color(card))


def _list_difference(prev_list, curr_list) -> list:
    """curr_list 相对 prev_list 新增的卡牌(按结构键做多重集差)。"""
    prev_counts = Counter(_card_key(c) for c in prev_list)
    seen = Counter()
    out = []
    for c in curr_list:
        k = _card_key(c)
        if seen[k] < prev_counts.get(k, 0):
            seen[k] += 1
        else:
            out.append(c)
    return out


def _newly_purchased(prev_state, state, idx) -> list:
    return _list_difference(prev_state.players[idx].purchased,
                            state.players[idx].purchased)


def _newly_reserved(prev_state, state, idx) -> list:
    return _list_difference(prev_state.players[idx].reserved,
                            state.players[idx].reserved)


# ===========================================================================
# 卡人 / restore:恰好差 1 个宝石(落在 cost 最大颜色,金币抵扣后总缺口==1)
# ===========================================================================
def _exactly_one_gem_away(player: PlayerState, card: Card) -> bool:
    """该玩家离买下 card 只差 1 个宝石,且这 1 个落在 cost 最大的颜色上。

    口径(已确认):按 bonus + 同色 token 抵扣后逐色缺口求和,再用金币抵扣;
    "总缺口(含金币抵扣后)== 1",且这最后 1 个能落在 cost 最大的颜色(该色仍有缺口)。
    """
    tokens = player.tokens
    bonuses = player.bonuses
    deficits = []
    for c in range(NUM_COLORS):
        d = int(card.cost[c]) - int(bonuses[c]) - int(tokens[c])
        deficits.append(max(0, d))

    gold = int(tokens[GOLD_INDEX])
    remaining = sum(deficits) - gold
    if remaining != 1:
        return False

    max_cost = max(int(card.cost[c]) for c in range(NUM_COLORS))
    if max_cost <= 0:
        return False
    # 最后 1 个未覆盖的宝石可安置在某个 cost 最大的颜色(该色仍有缺口)。
    for c in range(NUM_COLORS):
        if int(card.cost[c]) == max_cost and deficits[c] >= 1:
            return True
    return False


def _block_reward(prev_state: GameState, player_idx: int, demand: np.ndarray,
                  w_point: float, w_bonus: float,
                  newly_purchased: list, newly_reserved: list) -> float:
    """加和式卡人/restore 奖励(在 prev_state 上判定,按本步是否真的拿到该牌给奖励)。

    (A) 对手对"恰好一张且只有一张"好牌差 1 个宝石 → 我方买下或预留它 = 卡人。
    (B) 我方对"恰好一张且只有一张"好牌差 1 个宝石 → 我方预留它锁定 = 防卡 restore。
    奖励基数 = 该牌的边际价值估计(_card_marginal_value),与 Φ 同量纲。
    """
    me = prev_state.players[player_idx]
    opp = prev_state.players[1 - player_idx]
    face_up = _all_face_up(prev_state)

    acquired_keys = Counter(_card_key(c) for c in (newly_purchased + newly_reserved))
    reserved_keys = Counter(_card_key(c) for c in newly_reserved)

    total = 0.0

    # (A) 卡对手:对手差 1 的好牌恰好一张,且我方本步买/预留了它。
    opp_targets = [c for c in face_up
                   if _cost_performance(c, opp.bonuses) >= CP_THRESHOLD
                   and _exactly_one_gem_away(opp, c)]
    if len(opp_targets) == 1:
        X = opp_targets[0]
        if acquired_keys.get(_card_key(X), 0) > 0:
            val = _card_marginal_value(X, demand, w_point, w_bonus)
            total += BLOCK_SIGN * BLOCK_WEIGHT * val

    # (B) 防卡 restore:我方差 1 的好牌恰好一张,且我方本步预留了它。
    self_targets = [c for c in face_up
                    if _cost_performance(c, me.bonuses) >= CP_THRESHOLD
                    and _exactly_one_gem_away(me, c)]
    if len(self_targets) == 1:
        X = self_targets[0]
        if reserved_keys.get(_card_key(X), 0) > 0:
            val = _card_marginal_value(X, demand, w_point, w_bonus)
            total += BLOCK_SIGN * BLOCK_WEIGHT * val

    return total


# ===========================================================================
# Main reward function
# ===========================================================================
def compute_reward(
    prev_state: GameState,
    state: GameState,
    action: int,
    player_idx: int,
    gamma: float = SHAPING_GAMMA,
) -> float:
    """行动方的整形奖励。

    reward = γ·Φ(s') − Φ(s)  +  build_bonus  +  block/restore  +  terminal  +  penalties

    demand 与 w_point/w_bonus 在本次调用内冻结(取自 prev_state 的行动方),
    供 Φ(prev)/Φ(curr) 共用,消除翻牌噪声与买牌造成的权重跳变。
    """
    acting_prev: PlayerState = prev_state.players[player_idx]

    # 冻结 demand 与权重(整次调用一致)。
    demand = _demand_weights(prev_state, acting_prev.bonuses)
    n = len(acting_prev.purchased)
    w_point, w_bonus = _weights_for_n(n)

    # ------------------------------------------------------------------
    # 1. 终局:Φ(terminal)=0,收口项 = −Φ(prev)。
    # ------------------------------------------------------------------
    if state.game_over:
        if state.winner == player_idx:
            terminal = TERMINAL_WIN
        elif state.winner == (1 - player_idx):
            terminal = TERMINAL_LOSS
        else:
            terminal = TERMINAL_DRAW

        if APPLY_TERMINAL_SHAPING: 
            terminal += -_potential(prev_state, player_idx, demand, w_point, w_bonus)
        return float(terminal)

    # ------------------------------------------------------------------
    # 2. 势函数骨架(会望远镜求和)。
    # ------------------------------------------------------------------
    reward = (
        gamma * _potential(state, player_idx, demand, w_point, w_bonus)
        - _potential(prev_state, player_idx, demand, w_point, w_bonus)
    )

    # ------------------------------------------------------------------
    # 3. 加和式附加奖励(非势函数,刻意置于不变性保证之外)。
    # ------------------------------------------------------------------
    newly_purchased = _newly_purchased(prev_state, state, player_idx)
    newly_reserved = _newly_reserved(prev_state, state, player_idx)

    # 3a. 买到性价比≥0.5 的构筑牌。
    for card in newly_purchased:
        if _cost_performance(card, acting_prev.bonuses) >= CP_THRESHOLD:
            reward += BUILD_BONUS

    # 3b. 卡人 / 防卡 restore。
    reward += _block_reward(prev_state, player_idx, demand, w_point, w_bonus,
                            newly_purchased, newly_reserved)

    # ------------------------------------------------------------------
    # 4. 小额动作罚项(非势函数)。
    # ------------------------------------------------------------------
    if (0 <= action <= 14) or (45 <= action <= 49):
        if int(np.sum(prev_state.players[player_idx].tokens)) >= 10:
            reward += HOARDING_PENALTY
    if action == 50:
        reward += PASS_PENALTY
    reward += STEP_PENALTY

    return float(reward)
