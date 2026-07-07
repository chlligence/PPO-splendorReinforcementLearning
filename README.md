# Splendor (璀璨宝石) 强化学习训练项目 — 说明文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [模块详解](#3-模块详解)
   - [3.1 游戏核心 `splendor/`](#31-游戏核心-splendor)
   - [3.2 训练框架 `training/`](#32-训练框架-training)
   - [3.3 脚本 `scripts/`](#33-脚本-scripts)
   - [3.4 测试 `tests/`](#34-测试-tests)
4. [游戏规则实现对照](#4-游戏规则实现对照)
5. [观察空间设计](#5-观察空间设计)
6. [动作空间设计](#6-动作空间设计)
7. [奖励函数设计](#7-奖励函数设计)
8. [训练架构](#8-训练架构)
9. [环境依赖](#9-环境依赖)
10. [如何开始训练](#10-如何开始训练)
11. [硬件优化说明](#11-硬件优化说明)

---

## 1. 项目概述

本项目使用 **PPO（Proximal Policy Optimization）+ 自对弈（Self-Play）** 训练一个双人璀璨宝石（Splendor）游戏智能体。

- **算法**：Model-Free 策略梯度 PPO（来自 Stable-Baselines3）
- **动作掩码**：通过 `sb3-contrib` 的 `MaskablePPO` 支持，确保只探索合法动作
- **训练模式**：自对弈——每代从历史对手池中采样对手，迭代进化
- **硬件目标**：NVIDIA RTX 5070 Ti (16GB VRAM) + 24核 CPU + 32GB RAM

---

## 2. 目录结构

```
splendor_20260706/
│
├── cards_data.xlsx              # 原始卡牌数据（90张卡）
├── prompt.txt                   # 原始需求说明
├── requirements.txt             # Python依赖列表
├── 说明文档.md                   # 本文档
│
├── splendor/                    # 【游戏引擎核心】
│   ├── __init__.py
│   ├── constants.py             # 枚举、常量、游戏上限
│   ├── card.py                  # Card数据类、xlsx加载器
│   ├── game_state.py            # 游戏状态数据结构（纯数据，无逻辑）
│   ├── rules.py                 # 游戏规则引擎（纯函数）
│   ├── action_mask.py           # 合法动作掩码计算
│   ├── observation.py           # 观察向量构建（203维）
│   ├── reward.py                # 奖励塑形函数
│   └── env.py                   # Gymnasium环境主类
│
├── training/                    # 【训练框架】
│   ├── __init__.py
│   ├── config.py                # 超参数与路径配置
│   ├── feature_extractor.py     # 自定义神经网络特征提取器
│   ├── opponent_pool.py         # 对手池管理（ELO采样）
│   ├── self_play_env.py         # Self-Play环境包装器
│   ├── evaluate.py              # 对抗评估与ELO估算
│   └── self_play_loop.py        # 主训练循环
│
├── scripts/                     # 【入口脚本】
│   ├── verify_cards.py          # 卡牌数据验证
│   └── train.py                 # 训练启动入口
│
├── tests/                       # 【测试】
│   ├── __init__.py
│   ├── test_rules.py            # 规则逻辑单元测试（20个）
│   ├── test_action_mask.py      # 动作掩码测试（16个）
│   └── test_env.py              # 集成测试（1000局随机游戏）
│
├── checkpoints/                 # 【输出】模型保存目录
└── logs/                        # 【输出】TensorBoard 日志
```

---

## 3. 模块详解

### 3.1 游戏核心 `splendor/`

#### `splendor/constants.py` — 全局常量

定义所有游戏相关的枚举和数值上限。

| 定义 | 值 | 说明 |
|------|-----|------|
| `Gem` (IntEnum) | BLACK=0, WHITE=1, RED=2, BLUE=3, GREEN=4, GOLD=5 | 宝石颜色枚举 |
| `NUM_COLORS` | 5 | 基础宝石颜色数（不含黄金） |
| `MAX_GEMS_PER_COLOR` | 4 | 双人模式每种颜色宝石数 |
| `MAX_GOLD` | 5 | 黄金万能筹码数 |
| `MAX_RESERVED_CARDS` | 3 | 手牌上限 |
| `MAX_TOTAL_TOKENS` | 10 | 筹码持有上限 |
| `WINNING_POINTS` | 15 | 触发终局的分值阈值 |
| `CARDS_PER_LEVEL` | {1:40, 2:30, 3:20} | 每个等级的卡牌数 |
| `FACE_UP_PER_LEVEL` | 4 | 每个等级的明牌数 |
| `COMBO_3_DIFFERENT` | 10个三元组 | C(5,3) 所有取3不同颜色组合 |

#### `splendor/card.py` — 卡牌数据

**`Card`** — 不可变数据类（`@dataclass(frozen=True)`）：

```python
Card(card_id, level, bonus: Gem, points, cost: tuple[5])
```

**`load_cards(xlsx_path) -> dict[int, list[Card]]`** — 使用 `openpyxl` 解析 `cards_data.xlsx`，返回 `{1: [40张], 2: [30张], 3: [20张]}`。自动校验每个等级的卡牌数量。

**`get_card_summary(cards)`** — 生成卡牌数据库的可读摘要（分值分布、Bonus分布、费用范围）。

#### `splendor/game_state.py` — 游戏状态

**`PlayerState`** — 单玩家状态：

| 字段 | 类型 | 说明 |
|------|------|------|
| `tokens` | `np.ndarray(6,)` | 各色筹码数 [黑,白,红,蓝,绿,金] |
| `bonuses` | `np.ndarray(5,)` | 永久折扣（已购卡Bonus累计） |
| `reserved` | `list[Card]` | 手牌（最多3张，对手不可见内容） |
| `purchased` | `list[Card]` | 已购卡牌 |
| `points` | `int` | 总分 |
| `card_count` | `int` | 已购卡数（平局裁定用） |

**`GameState`** — 完整游戏状态（纯数据容器，逻辑在 rules.py）：

| 字段 | 说明 |
|------|------|
| `face_up: dict[int, list[Card\|None]]` | 明牌区，每级4张 |
| `decks: dict[int, list[Card]]` | 牌堆，末尾为牌堆顶 |
| `gems_available: np.ndarray(6,)` | 场上可用宝石 |
| `players: tuple[PlayerState, PlayerState]` | 双玩家 |
| `current_player: int` | 当前回合玩家 (0=P0先手, 1=P1后手) |
| `final_round_flag: bool` | 终局标记 |
| `final_round_player: int\|None` | 触发终局的玩家 |
| `game_over: bool` | 游戏结束 |
| `winner: int\|None` | 胜者 (0/1/None=平局) |

**`clone_state(state)`** — 深拷贝整个游戏状态，用于奖励计算时保存"动作前"快照。

#### `splendor/rules.py` — 规则引擎

纯函数实现，所有操作原地修改 `GameState`。

**`execute_action(state, player_idx, action)`** — 动作分发器：

| 动作索引 | 动作类型 | 核心逻辑 |
|----------|----------|----------|
| 0–9 | 取3不同色宝石 | 从10个组合中取3色各1枚 |
| 10–14 | 取2同色宝石 | 取2枚同色（需≥4存量） |
| 15–26 | 留明牌 | 从12个明牌位选1，获1金（若有） |
| 27–29 | 留牌堆顶 | 从3个等级牌堆盲抽 |
| 30–41 | 购买明牌 | 支付成本（筹码+Bonus+黄金补差） |
| 42–44 | 购买手牌 | 购买已保留的卡 |
| 45–49 | 取1枚宝石 | 取单色1枚 |
| 50 | 跳过 | 无操作（死锁兜底） |

**`can_afford(tokens, bonuses, cost) -> bool`** — 支付能力检查。用黄金万能筹码1:1补足差额。

**购牌流程** (`_purchase_card`)：
1. 计算有效成本 `max(0, 牌面成本 - Bonus折扣)`
2. 优先用对应色筹码支付
3. 差额用黄金补齐
4. 卡牌移入 `purchased`，增加 Bonus 和分值
5. 面朝上购买时从牌堆补1张明牌

**终局判定** (`check_game_end`)：
```
若玩家分数 ≥ 15:
  若先手(P0)触发 → 标记final_round_flag，P1获最后一次回合
  若后手(P1)触发 → 游戏立即结束（双方回合数相等）
若final_round_flag已触发且轮到触发者 → 游戏结束
```

**弃牌逻辑** (`_auto_discard_if_needed`)：回合结束筹码 > 10 时自动弃牌。优先级（价值从低到高弃）：(1)数量最多的颜色 → (2)颜色固定顺序 BLACK>WHITE>RED>BLUE>GREEN → (3)黄金最后（万能筹码，价值最高）。确定性策略，智能体可学习适应。

#### `splendor/action_mask.py` — 动作掩码

**`get_action_mask(state, player_idx) -> np.ndarray[bool, 51]`** — 计算当前状态下所有合法动作：

| 检查条件 | 示例 |
|----------|------|
| 取3不同色 → 3色各有≥1枚 | board gems [3,4,4,4,4,5] → 全部10种合法 |
| 取2同色 → 该色≥4枚 | board gems 某色=4 → 该色合法 |
| 取1枚 → 该色≥1枚 | 各色有存量则合法 |
| 留牌 → 手牌<3 **且** 目标位有牌/牌堆非空 | 手牌满→全部留牌动作屏蔽 |
| 购牌 → `can_afford()` 返回 True | 筹码不足→屏蔽 |
| 跳过 → **始终合法** | 兜底防死锁 |

**`get_action_description(action) -> str`** — 动作索引→可读描述（调试用）。

#### `splendor/observation.py` — 观察向量

**`build_observation(state, player_idx) -> np.ndarray[float32, 203]`** — 构建归一化观察向量（所有值∈[0,1]）。

**始终从"即将行动者"视角编码**：当前玩家=自我，对手=对方。

```
观察向量布局（203维）：

┌─ 公开信息（141维）────────────────────────────────
│  明牌: 12张 × 11维 = 132维
│    每张卡: bonus_onehot(5) + points/15(1) + cost/7(5)
│  牌堆剩余: 3维  L1/40, L2/30, L3/20
│  场上宝石: 6维  5色/4 + 黄金/5
│
├─ 自我信息（46维）─────────────────────────────────
│  筹码: 6维  /10
│  Bonus折扣: 5维  /15
│  手牌(保留): 3×11=33维  (空位=全零向量)
│  分数: 1维  /15
│  卡牌数: 1维  /15
│
├─ 对手信息（14维）─────────────────────────────────
│  筹码: 6维  /10
│  Bonus折扣: 5维  /15
│  手牌数量: 1维  /3    ← 内容不可见（非完美信息）
│  分数: 1维  /15
│  卡牌数: 1维  /15
│
└─ 全局标记（2维）─────────────────────────────────
   当前回合标记: 1维  (0或1)
   终局标记: 1维  (final_round_flag)
```

#### `splendor/reward.py` — 奖励塑形

**`compute_reward(prev_state, state, action, player_idx) -> float`**：

| 事件 | 奖励值 | 权重比率 | 目的 |
|------|--------|----------|------|
| **胜利** | **+1.0** | ★ 主导 | 终局激励 |
| **失败** | **-1.0** | ★ 主导 | 终局激励 |
| 平局 | 0.0 | | |
| 购牌得分 | +0.05 × 分值 | 微小 | 鼓励买牌进步 |
| 率先达到15分 | +0.3 | 中等 | 奖励优势局面 |
| 筹码囤积(>8) | -0.01 × (数-8) | 微小 | 抑制无效囤积 |
| 纯Bonus买牌 | +0.02 | 微小 | 奖励引擎建设 |
| 满筹时买牌 | -0.03 | 微小 | 抑制低效行为 |

> 终局奖 ±1 完全压倒塑形奖，确保智能体优化目标始终是"赢棋"。

#### `splendor/env.py` — Gymnasium环境

**`SplendorEnv`** — 实现标准 `gym.Env` 接口：

```python
env = SplendorEnv(cards)
obs, info = env.reset()                        # info["action_mask"]
obs, reward, terminated, truncated, info = env.step(action)
```

关键实现细节：
- 观察空间：`Box(203,)`，全 float32，归一化到 [0,1]
- 动作空间：`Discrete(51)`，合法动作通过 `info["action_mask"]` 传递
- 每步执行：`clone_state` → `execute_action` → `check_game_end` → `compute_reward` → 切换玩家
- 渲染支持：`render()` 返回 ASCII 文本棋盘

---

### 3.2 训练框架 `training/`

#### `training/config.py` — 超参数配置

**PPO 参数**（针对 5070 Ti 16GB 调优）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_steps` | 2048 | 每个环境每轮采样步数 |
| `batch_size` | 512 | 小批量大小 |
| `n_epochs` | 10 | 每次更新的 PPO epoch 数 |
| `learning_rate` | 3e-4 | 学习率 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 折扣因子 / GAE 参数 |
| `ent_coef` | 0.01 (初始) | 熵系数（探索激励） |

**Self-Play 参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_envs` | 20 | 并行环境数（占~85% CPU） |
| `generations` | 50 | 总代数 |
| `steps_per_generation` | 500,000 | 每代训练步数 |
| `opponent_pool_size` | 20 | 对手池最大容量 |
| `eval_games` | 200 | 每轮 ELO 评估局数 |
| `ent_start / ent_end` | 0.05 → 0.005 | 熵系数退火（前10代） |

**内存预算**：模型 ~4MB + 缓冲区 ~35MB + 梯度 ~8MB = **~50MB GPU 显存**，剩余 ~15.95GB 可用于增大 Batch/模型。

#### `training/feature_extractor.py` — 神经网络

```
SplendorFeatureExtractor (BaseFeaturesExtractor)
  输入: Box(203,)
  架构: 203 → Linear(512)+LayerNorm+ReLU
             → Linear(512)+LayerNorm+ReLU
             → Linear(512)+LayerNorm+ReLU
             → Linear(256)+ReLU
  参数: ~760K
  初始化: 正交初始化 (SB3 标准)
```

**策略头** (Actor)：`256 → Linear(512)+ReLU → Linear(512)+ReLU → Linear(51)` （动作 logits）

**价值头** (Critic)：`256 → Linear(512)+ReLU → Linear(512)+ReLU → Linear(1)` （状态价值）

**总参数量**：~1M，约 4MB GPU 显存。

#### `training/opponent_pool.py` — 对手池

**`PoolEntry`** — 池条目：`path, generation, elo, win_rate_vs_prev`

**`OpponentPool`** — 对手池管理：

- **采样策略**（可配置概率）：
  - 50% → 最新对手（和当前水平最接近）
  - 40% → ELO-Softmax 加权（偏好强对手）
  - 10% → 均匀随机（保持多样性）
- **满员处理**：移除 ELO 最低的条目
- **持久化**：`save_index()` / `load_index()` 保存/恢复为 JSON

#### `training/self_play_env.py` — Self-Play 环境包装器

**核心思想**：将双人博弈包装成"单智能体"环境，对手自动操控。

```
SelfPlayEnv.step(agent_action):
  1. 执行智能体动作 → inner.step(agent_action)
  2. while 轮到对手 且 游戏未结束:
       对手观察 → opponent_model.predict() → inner.step(opp_action)
  3. 返回 (obs, reward, terminated, info)
  // 对手的奖励被丢弃，只保留智能体的迁移数据
```

**`make_env_fn(cards_path, opponent_path, ...)`** — SubprocVecEnv 工厂函数。每个子进程独立加载对手模型（传路径而非对象，避免 pickle 问题）。

#### `training/evaluate.py` — 评估

**`evaluate_head_to_head(model_a, model_b, cards, num_games)`** — 对抗评估：
- 双方各当 P0 一半对局（消除先手优势）
- 返回胜/负/平统计

**`estimate_elo(model, pool, cards, num_games)`** — ELO 估算：
- 和池中 ELO 最高3个对手各打 N 局
- 用 ELO 公式更新评分 (K=32)

#### `training/self_play_loop.py` — 主训练循环

**`run_self_play()`** — 完整自对弈流程：

```
For generation in 0..50:
  1. 从对手池采样对手（或随机对手，第0代）
  2. 创建 20 个并行 SelfPlayEnv
  3. 创建/加载 MaskablePPO 模型
  4. 退火更新熵系数 ent_coef
  5. model.learn(total_timesteps=500K)
  6. 保存 checkpoint
  7. 评估 ELO 并加入对手池
  8. 清理环境，进入下一代
```

**`train_vs_random_baseline()`** — 对抗随机对手的基线训练（无需对手池，方便调试）。

---

### 3.3 脚本 `scripts/`

#### `scripts/verify_cards.py`

验证 `cards_data.xlsx` 的加载和完整性。运行后打印：
- 每级卡牌数量、分值分布、Bonus分布、费用范围
- 完整性校验（Bonus合法、Cost非负等）
- 抽样展示

```bash
python scripts/verify_cards.py
```

#### `scripts/train.py` — 训练入口

```bash
# 完整 Self-Play 训练（50代）
python scripts/train.py

# 基线训练（对抗随机对手，更简单）
python scripts/train.py --baseline

# 快速测试（5代、4环境、50K步/代）
python scripts/train.py --test

# 从检查点恢复
python scripts/train.py --resume checkpoints/agent_gen_5.zip

# 使用 CPU 训练
python scripts/train.py --device cpu
```

---

### 3.4 测试 `tests/`

#### `tests/test_rules.py` — 规则单元测试（20个）

| 测试类 | 测试内容 |
|--------|----------|
| `TestCanAfford` (6) | 简单支付、不够付、Bonus折扣、黄金万能、组合支付、高价牌 |
| `TestBuyCard` (3) | 买牌扣币、黄金替代、Bonus折扣生效 |
| `TestTakeGems` (2) | 取3不同色、取2同色 |
| `TestReserve` (3) | 留牌获金、金空仍可留、手牌满禁止 |
| `TestGameEnd` (3) | P0触发终局、P1触发即结束、平局裁定 |
| `TestTokenOverflow` | 筹码超10自动弃牌 |
| `TestCloneState` (2) | 深拷贝独立性 |

#### `tests/test_action_mask.py` — 动作掩码测试（16个）

| 测试类 | 测试内容 |
|--------|----------|
| `TestInitialMask` (5) | 初始全合法、买牌初始不可用、留牌可用 |
| `TestGemExhaustion` (3) | 存量3时禁取2、存量4时可取2、零存禁含该色组合 |
| `TestReserveFull` | 手牌满全屏蔽 |
| `TestBuyMask` (3) | 有筹时开放、无筹时屏蔽、手牌可买 |
| `TestMaskShape` (3) | 形状正确、始终有合法动作(Pass)、取1动作 |

#### `tests/test_env.py` — 集成测试

运行 1000 局随机合法动作游戏，验证：
- 0 负数 Token 错误
- 0 筹码溢出错误
- 0 资源池负数错误
- 100% 游戏正常结束（无死锁）
- 动作分布统计

```bash
python tests/test_env.py
```

---

## 4. 游戏规则实现对照

| # | 官方规则 | 实现位置 | 实现方式 |
|---|----------|----------|----------|
| 1 | 5色各4枚、黄金5枚 | `constants.py` | `INITIAL_GEMS = [4,4,4,4,4,5]` |
| 2a | 取3不同色宝石 | `rules.py:_take_3_different` | C(5,3)=10 组合，各取1 |
| 2b | 取2同色（需≥4存量） | `action_mask.py:10-14` | `mask[10+c] = gems[c]>=4` |
| 3 | 留牌上限3张+获金 | `rules.py:_add_reserved_card` | `len(reserved)<3` 检查+金余量检查 |
| 4 | 筹码上限10枚 | `rules.py:_auto_discard_if_needed` | 超10自动弃，优先最高色→固定顺序→金最后 |
| 5 | Bonus永久折扣 | `rules.py:_purchase_card` | `effective_cost = max(0, cost-bonus)` |
| 6a | 15分触发终局 | `rules.py:check_game_end` | P0触发→P1最后一动；P1触发→立即结束 |
| 6b | 平局少牌者胜 | `rules.py:_determine_winner` | points→card_count→draw |
| 7 | 无贵族卡 | 整体设计 | 未实现 Noble 相关逻辑 |

---

## 5. 观察空间设计

### 设计原则

1. **平面化**：203维 Box 空间，兼容 SB3，无 Dict 开销
2. **归一化**：全部特征 ∈ [0,1]，提高训练稳定性
3. **对称视角**：始终以"即将行动者"为 self，对手为 opponent，网络学到对称策略
4. **非完美信息**：对手手牌只暴露数量（count=1维），不暴露内容（真实规则中保留的卡是面朝下的）

### 归一化参考值

| 特征 | 除数 | 原因 |
|------|------|------|
| 牌面分 | 15 | 与胜利阈值一致 |
| 单色成本 | 7 | 数据集中 L3 某卡单色成本最高为7 |
| 玩家筹码 | 10 | 筹码持有上限 |
| Bonus | 15 | 合理上限（每色最多18张该色Bonus卡） |
| 牌堆剩余 | 40/30/20 | 各等级起始数量 |
| 场上宝石 | 4/5 | 各色/黄金初始数量 |

---

## 6. 动作空间设计

### 动作编码（Discrete(51)）

| 索引 | 数量 | 类型 | 说明 |
|------|------|------|------|
| 0–9 | 10 | 🟢 取3不同色 | C(5,3)=10 种颜色组合 |
| 10–14 | 5 | 🟢 取2同色 | 5 种颜色各一 |
| 15–26 | 12 | 🟡 留面朝上牌 | 3级 × 4位置 |
| 27–29 | 3 | 🟡 留牌堆顶 | 3 个等级盲抽 |
| 30–41 | 12 | 🔴 买面朝上牌 | 3级 × 4位置 |
| 42–44 | 3 | 🔴 买手牌 | 3 个保留槽位 |
| 45–49 | 5 | 🟢 取1枚 | 5 种颜色各一 |
| 50 | 1 | ⚪ 跳过 | 始终合法，防止死锁 |

### 死锁问题的解决

在极端情况下（筹码均枯竭、手牌满、无牌可买），玩家可能无合法动作。第 50 号动作 **"跳过"** 始终合法，作为兜底。在实际对局中极其罕见，但确保环境永远不会陷入死循环。

---

## 7. 奖励函数设计

### 核心原则

**辅助奖励不能支配终局奖励**。所有塑形奖励乘以小系数（0.01–0.3），终局 ±1 始终是主要学习信号。

### 奖励信号总结

```
终局 ±1.0         ████████████████████████████████  主导信号
率先15分 +0.3      █████████                          次要信号
购牌 +0.05×分      ██                                 微弱引导
囤积 -0.01×溢      █                                  抑制信号
纯Bonus +0.02      █                                  鼓励信号
满筹买牌 -0.03     █                                  抑制信号
```

---

## 8. 训练架构

### Self-Play 流程

```
┌──────────────────────────────────────────────────────┐
│                    第 N 代训练                        │
│                                                      │
│  ┌──────────┐    采样对手    ┌──────────────┐        │
│  │ 对手池    │ ────────────→ │ 对手模型      │        │
│  │ (≤20个)  │   50%最新      │ (MaskablePPO) │        │
│  │          │   40%ELO加权   └──────┬───────┘        │
│  │          │   10%随机             │                │
│  └──────────┘                      ▼                │
│                    ┌────────────────────────┐       │
│                    │   SelfPlayEnv × 20     │       │
│                    │   (SubprocVecEnv)       │       │
│                    │                        │       │
│                    │  智能体(P0) vs 对手(P1)  │       │
│                    │  仅收集智能体的迁移数据   │       │
│                    └───────────┬────────────┘       │
│                                ▼                    │
│                    ┌────────────────────────┐       │
│                    │  MaskablePPO.learn()   │       │
│                    │  500K steps             │       │
│                    └───────────┬────────────┘       │
│                                ▼                    │
│                    ┌────────────────────────┐       │
│                    │  ELO评估 → 加入对手池   │       │
│                    └────────────────────────┘       │
└──────────────────────────────────────────────────────┘
```

### 熵系数退火

```
ent_coef
  0.05 ┤╲
       │ ╲
       │  ╲
  0.02 │   ╲_____________
       │                  
 0.005 ┤──────────────────
       0    5    10    ...    50   generation
       
前10代从0.05线性退火到0.005，之后保持恒定。
早期高熵鼓励探索，后期低熵专注优化。
```

---

## 9. 环境依赖

### 硬件

| 组件 | 规格 | 用途 |
|------|------|------|
| GPU | NVIDIA RTX 5070 Ti 16GB | 神经网络前向/反向传播 |
| CPU | 24核 24线程 | 20并行环境采样 |
| RAM | 32 GB | 多进程环境 + 经验缓冲 |

### 虚拟环境（推荐）

项目根目录已创建 `.venv/` 虚拟环境。**请先激活再安装依赖**，避免污染系统 Python：

```bash
# 激活虚拟环境（Git Bash / 终端）
source .venv/Scripts/activate

# 或 Windows CMD:
.venv\Scripts\activate.bat

# 或 PowerShell:
.venv\Scripts\Activate.ps1
```

### 软件

> **注意**：以下"已安装"指的是之前装在**系统 Python** 中的包。建议激活 `.venv` 后重新安装所有依赖（见[如何开始训练](#10-如何开始训练)）。

#### ✅ 已安装（系统 Python 中）

| 包名 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.10.11 | 运行环境 |
| **numpy** | 2.2.6 | 数值计算、数组操作 |
| **openpyxl** | 3.1.5 | 解析 cards_data.xlsx |
| **gymnasium** | 0.29.1 | RL 环境标准接口 |
| **pytest** | 9.1.1 | 单元测试框架 |
| **tensorboard** | 2.21.0 | 训练日志可视化 |
| cloudpickle | 3.1.2 | 序列化（gymnasium 依赖） |
| grpcio | 1.82.0 | gRPC（tensorboard 依赖） |
| protobuf | 7.35.1 | 协议缓冲（tensorboard 依赖） |
| absl-py | 2.5.0 | Google 通用库（tensorboard 依赖） |
| Markdown | 3.10.2 | Markdown 处理（tensorboard 依赖） |
| Werkzeug | 3.1.8 | WSGI 工具（tensorboard 依赖） |
| packaging | 26.2 | 包版本解析 |
| pillow | 12.3.0 | 图像处理 |
| typing_extensions | 4.16.0 | 类型提示扩展 |
| Farama-Notifications | 0.0.6 | Gymnasium 通知 |

#### ❌ 需手动安装（训练必需）

以下包下载超时或需要你手动安装：

```bash
# 1. PyTorch（核心深度学习框架，~2.5GB）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Stable-Baselines3（PPO算法实现）
pip install stable-baselines3==2.3.0

# 3. SB3-Contrib（MaskablePPO 动作掩码支持）
pip install sb3-contrib==2.3.0
```

安装完成后验证：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from sb3_contrib import MaskablePPO; print('MaskablePPO OK')"
```

### 完整 requirements.txt

```
torch>=2.0
stable-baselines3==2.3.0
sb3-contrib==2.3.0
gymnasium==0.29.1
openpyxl>=3.1
numpy>=1.24
tensorboard>=2.13
```

---

## 10. 如何开始训练

### 第一步：激活虚拟环境并安装依赖

```bash
# 进入项目目录
cd D:\ReinforcementLearning\SPLENSOR\splendor_20260706

# 激活虚拟环境
source .venv/Scripts/activate

# 安装所有依赖
pip install numpy openpyxl gymnasium==0.29.1 pytest tensorboard
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3==2.3.0 sb3-contrib==2.3.0
```

### 第二步：验证环境

```bash
# 验证卡牌数据加载
python scripts/verify_cards.py

# 运行单元测试（需先安装 pytest: pip install pytest）
python -m pytest tests/ -v

# 运行集成测试（100局随机游戏）
python tests/test_env.py
```

### 第三步：快速测试训练

```bash
# 快速测试模式：5代、4环境、50K步/代、约5-10分钟
python scripts/train.py --test
```

### 第四步：正式训练

```bash
# 完整 Self-Play 训练（50代，预计数天）
python scripts/train.py

# 或先跑基线（对抗随机对手，更简单）
python scripts/train.py --baseline

# 从检查点恢复训练
python scripts/train.py --resume checkpoints/agent_gen_10.zip
```

### 监控训练

```bash
# 启动 TensorBoard 查看训练曲线
tensorboard --logdir logs/
# 浏览器打开 http://localhost:6006
```

关注的指标：
- `rollout/ep_rew_mean` — 平均回合奖励（应随代数上升）
- `train/policy_gradient_loss` — 策略梯度损失
- `train/value_loss` — 价值函数损失
- `train/entropy_loss` — 熵（不应过早趋于零）

---

## 11. 硬件优化说明

### CPU 并行采样

`SubprocVecEnv` 启动 20 个独立 Python 子进程，每个运行一个 `SelfPlayEnv` 实例。24 核 CPU 留出 4 核给系统和其他任务。

### GPU 显存利用

当前配置下 GPU 显存占用约 **50MB**（模型 4MB + 缓冲区 35MB + 梯度 8MB），16GB 中剩余 ~15.95GB。这意味着：

- **可以增大模型**：加宽隐藏层（512→1024）、加深网络
- **可以增大 Batch**：`n_steps` 从 2048 增至 8192，`batch_size` 从 512 增至 2048
- **可以增大并行**：`n_envs` 从 20 增至 32（若 CPU 允许）

调整这些参数可参看 `training/config.py`。

### 内存使用

20 个环境进程 × ~20KB/状态 + Python 进程开销 ≈ **< 500MB**，远在 32GB 限制内。

---

## 附录：常见问题

**Q: 训练时 GPU 利用率低？**
A: 增大 `n_envs` 或 `n_steps` 来喂更多数据给 GPU。检查 `training/config.py` 中的参数。

**Q: 智能体只学会刷筹码不买牌？**
A: 增大 `ent_coef` 初始值（0.05→0.1），或增加购牌塑形奖励权重。

**Q: 对手池里全是相似的对手？**
A: 增大 `random_opponent_prob`（0.10→0.20），增加随机对手采样概率。

**Q: 某代突然变差？**
A: 正常现象——Self-Play 中对手策略变化会导致暂时震荡，通常几代后恢复。
