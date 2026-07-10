# Splendor (璀璨宝石) 强化学习训练项目 — 说明文档

---
## 快速开始

cd 

激活虚拟环境
 或Windows CMD:
.venv\Scripts\activate.bat

#或 PowerShell:
.venv\Scripts\Activate.ps1

#安装所有依赖
pip install -r requirements.txt




**checkpoints中的50个zip文件是训练好的模型，通过修改/webapp/server.py:23行中的CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "agent_latest.zip")最后一个参数即可选择加载不同代数的模型（这里显示的是最新代，50代的模型）。

启动server.py后，访问127.0.0.1:5000开始ui界面璀璨宝石人机对战。




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
    - [监控训练](#监控训练)
    - [Checkpoint 文件详解](#checkpoint检查点文件详解)
11. [硬件优化说明](#11-硬件优化说明)
12. [训练诊断与修复记录](#12-训练诊断与修复记录)

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
├── requirements.txt             # Python依赖列表
├── README.md                    # 本文档
│
├── splendor/                    # 【游戏引擎核心】
│   ├── __init__.py
│   ├── constants.py             # 枚举、常量、游戏上限
│   ├── card.py                  # Card数据类、xlsx加载器
│   ├── game_state.py            # 游戏状态数据结构（纯数据，无逻辑）
│   ├── rules.py                 # 游戏规则引擎（纯函数）
│   ├── action_mask.py           # 合法动作掩码计算
│   ├── observation.py           # 观察向量构建（203维）
│   ├── reward.py                # 奖励塑形函数（战略势能函数）
│   └── env.py                   # Gymnasium环境主类
│
├── training/                    # 【训练框架】
│   ├── __init__.py
│   ├── config.py                # 超参数与路径配置
│   ├── feature_extractor.py     # 自定义神经网络特征提取器
│   ├── opponent_pool.py         # 对手池管理（ELO采样、按代数取最新）
│   ├── self_play_env.py         # Self-Play环境包装器（含终局奖励回传）
│   ├── evaluate.py              # 对抗评估、ELO估算、逐代评估调度
│   └── self_play_loop.py        # 主训练循环
│
├── scripts/                     # 【入口脚本】
│   ├── verify_cards.py          # 卡牌数据验证
│   ├── train.py                 # 训练启动入口
│   └── rebuild_pool_index.py    # 离线重建 pool_index.json（用修复后的评估逻辑重跑已有 checkpoint）
│
├── tests/                       # 【测试】
│   ├── __init__.py
│   ├── test_rules.py            # 规则逻辑单元测试（20个）
│   ├── test_action_mask.py      # 动作掩码测试（16个）
│   └── test_env.py              # 集成测试（1000局随机游戏）
│
├── checkpoints/                 # 【输出】模型检查点保存目录
│   ├── agent_gen_N/              #   第 N 代模型文件夹（含 6 个内部文件）
│   ├── agent_gen_N.zip           #   第 N 代模型 ZIP 存档（~19 MB）
│   ├── agent_latest/             #   最新代模型文件夹副本
│   ├── agent_latest.zip          #   最新代模型 ZIP 副本
│   └── pool_index.json           #   对手池索引（ELO、胜率、代数、数据来源标注）
├── logs/                        # 【输出】TensorBoard 日志
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

#### `splendor/reward.py` — 奖励塑形（基于战略势能函数）

**核心理念**：构建综合"战略势能函数"评估玩家资产价值，每步奖励 = 势能差值变化 × 缩放系数。

**`get_strategic_potential(player, bonuses_array) -> float`** — 七维资产评分：

| 维度 | 权重 | 说明 |
|------|------|------|
| ① 基础分 | `points × 1.0` | 终极目标，权重最高 |
| ② 基建分 | `total_bonuses × 0.25` | 每张已购卡提供永久折扣 |
| ③ 同色深挖奖励 | `max_single_bonus × 0.08`（≥3时触发） | 单色Bonus质变形成战略威慑 |
| ④ 看板契合度 | `Σ(bonus[c] × demand[c]) × 0.15` | 资产颜色对齐版面高分牌需求 |
| ⑤ 筹码分 | `tokens[:5] × 0.02` | 购买力储备 |
| ⑥ 黄金分 | `gold × 0.025` | 万能筹码高弹性 |
| ⑦ 手牌分 | `reserve × 0.02`（≤2时） | 预留策略价值，满3扣分 |

**终局奖励**（始终主导）：

| 结局 | 奖励 |
|------|------|
| 胜利 | **+1.0** |
| 失败 | **-1.0** |
| 平局 | 0.0 |

**其它引导**：

| 机制 | 奖励/惩罚 | 目的 |
|------|-----------|------|
| 势能差值变化 | `0.05 × Δmargin` | 密集信号，零和趋势 |
| 版面动态需求分析 | 融入势能函数 | 引导买牌对齐高分牌颜色需求 |
| 满筹拿宝石 | `-0.04` | 防止无效刷筹码导致强制弃牌 |
| 纯Bonus白嫖买牌 | `+0.02` | 鼓励引擎建设效率 |

> 终局 ±1 完全压倒塑形奖，确保智能体优化目标始终是"赢棋"。

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

**PPO 参数**（针对 5070 Ti 16GB 调优，为降低 `approx_kl`/`clip_fraction` 已从早期版本收紧）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_steps` | 1024 | 每个环境每轮采样步数 |
| `batch_size` | 256 | 小批量大小 |
| `n_epochs` | 4 | 每次更新的 PPO epoch 数（降低数据重复利用，减少过拟合当前对手） |
| `learning_rate` | 5e-5 → 1e-5 | **代际余弦退火**（见下方"学习率退火"），非恒定值 |
| `gamma` / `gae_lambda` | 0.99 / 0.92 | 折扣因子 / GAE 参数 |
| `clip_range` | 0.15 | PPO 裁剪范围（收紧信任域） |
| `ent_coef` | 0.01 (初始，被退火覆盖) | 熵系数（探索激励） |
| `vf_coef` | 0.75 | 价值函数损失权重（增强 Critic 学习） |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |

**Self-Play 参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_envs` | 20 | 并行环境数（占~85% CPU） |
| `generations` | 50 | 总代数 |
| `steps_per_generation` | 500,000 | 每代训练步数 |
| `opponent_pool_size` | 20 | 对手池最大容量 |
| `eval_games` | 100 | 到 `eval_interval_generations` 间隔时的完整 ELO 评估局数 |
| `eval_interval_generations` | 5 | 每隔 N 代跑一次完整（昂贵）评估 |
| `cheap_eval_games` | 15 | **每一代**都会跑的轻量胜率/ELO 检查局数（match-pairs），避免非评估代完全没有真实信号 |
| `cheap_elo_k` | 16.0 | 轻量评估用的 ELO K 因子（样本少，取值比完整评估的 K=32 更保守） |
| `ent_start / ent_end` | 0.08 → 0.01 | 熵系数退火（前30代） |

> 关于 `eval_games` / `eval_interval_generations` / `cheap_eval_games`：早期版本存在一个 Bug——非评估代会直接写死 `win_rate_vs_prev=0.5`、`elo=best_elo+10`，导致 `pool_index.json` 里的数据大部分是伪造的（现象：`win_rate_vs_prev` 永远不变但 `elo` 单调上升）。现已修复为「每代真实轻量评估 + 到间隔时完整评估覆盖」，详见[训练诊断与修复记录](#12-训练诊断与修复记录)。

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

**`PoolEntry`** — 池条目：`path, generation, elo, win_rate_vs_prev, elo_source, win_rate_source`

- `win_rate_vs_prev` 为 `Optional[float]`：第 0 代没有上一代对手可比，值为 `None`（而非伪造的 0.5）。
- `elo_source` / `win_rate_source`：数据来源标注，取值 `"full_eval"`（完整评估）/ `"incremental_vs_latest"` 或 `"cheap_vs_latest"`（每代轻量评估）/ `"baseline"`（第0代，无历史可比）。用于让 `pool_index.json` 里的每一个数字都可追溯，不再有"凭空写死"的占位值。

**`OpponentPool`** — 对手池管理：

- **采样策略**（可配置概率）：
  - 50% → 最新对手（`get_latest_entry()`，按 `generation` 字段显式取值，与列表顺序无关——和当前水平最接近）
  - 40% → ELO-Softmax 加权（偏好强对手）
  - 10% → 均匀随机（保持多样性）
- **满员处理**：用 `min()+remove()` 移除 ELO 最低的条目，**不会**打乱 `self.entries` 的插入顺序（早期版本用 `sort()+pop(0)` 实现，会永久按 ELO 重排整个列表，导致"最新对手"语义被悄悄破坏——详见[训练诊断与修复记录](#12-训练诊断与修复记录)）。
- **持久化**：`save_index()` / `load_index()` 保存/恢复为 JSON，向后兼容读取不含新字段的旧 `pool_index.json`。

#### `training/self_play_env.py` — Self-Play 环境包装器

**核心思想**：将双人博弈包装成"单智能体"环境，对手自动操控。

```
SelfPlayEnv.step(agent_action):
  1. 执行智能体动作 → inner.step(agent_action)
  2. while 轮到对手 且 游戏未结束:
       对手观察 → opponent_model.predict() → inner.step(opp_action)
  3. 若对手的某一步结束了游戏：
       agent 收到的 reward = -(对手那一步的 reward)
       （reward.py 的终局值是零和的：+1/-1/0，对手视角取负即 agent 视角）
     否则：agent 收到自己那一步的 reward（对手中间步的塑形奖励仍被丢弃）
  4. 返回 (obs, reward, terminated, info)
```

> 早期版本无条件丢弃对手所有步的 reward，包括终局那一步——导致「对手最后一步获胜/agent 输棋」这种最常见的输局场景下，agent 完全学不到真实的 -1 终局信号。现已修复，详见[训练诊断与修复记录](#12-训练诊断与修复记录)。

**`make_env_fn(cards_path, opponent_path, ...)`** — SubprocVecEnv 工厂函数。每个子进程独立加载对手模型（传路径而非对象，避免 pickle 问题）。

#### `training/evaluate.py` — 评估

**`evaluate_head_to_head(model_a, model_b, cards, num_games)`** — 对抗评估：
- 双方各当 P0 一半对局（消除先手优势）
- 返回胜/负/平统计

**`estimate_elo(model, pool, cards, num_games, prior_elo=None)`** — ELO 估算：
- 和池中 ELO 最高2个对手各打 N 局
- 用 ELO 公式更新评分 (K=32)
- `prior_elo`：从 agent 自己上一次的已知 ELO 继续更新，而不是每次都从 1200 基线重新算——避免真实评估值和其他代次的数值不在同一量纲上（早期版本每次都重置为 1200，详见[训练诊断与修复记录](#12-训练诊断与修复记录)）

**`evaluate_generation(agent_model, pool, cards_path, generation, last_known_elo, ...)`** — 每代评估调度（被训练主循环和离线重建脚本 `scripts/rebuild_pool_index.py` 共用）：
- 每一代都用少量对局（`cheap_eval_games`）对最新对手做一次真实的轻量评估，得到 `win_rate_vs_prev` 和一次小幅 ELO 增量更新
- 到 `eval_interval_generations` 间隔时，额外跑一次完整的 `estimate_elo()` + 100 局 `evaluate_head_to_head()`，用更可靠的结果覆盖当代的轻量估计
- 返回值带 `elo_source`/`win_rate_source` 标签，写入 `PoolEntry`

#### `training/self_play_loop.py` — 主训练循环

**`run_self_play()`** — 完整自对弈流程：

```
For generation in 0..50:
  1. 从对手池采样对手（或随机对手，第0代）
  2. 创建 20 个并行 SelfPlayEnv
  3. 创建/加载 MaskablePPO 模型；按代数重算学习率并重建 lr_schedule
  4. 退火更新熵系数 ent_coef
  5. model.learn(total_timesteps=500K)
  6. 保存 checkpoint
  7. evaluate_generation()：每代轻量评估 + 到间隔时完整评估，更新 last_known_elo
  8. 加入对手池，保存 pool_index.json，清理环境，进入下一代
```

> 第 3 步的"重建 lr_schedule"：`compute_learning_rate(generation)` 计算的余弦退火值（5e-5→1e-5）早期版本只是重新赋值 `agent_model.learning_rate` 这个属性，但 SB3 实际读取学习率用的是初始化时生成的 `lr_schedule` 闭包，重新赋值不会让它生效——导致整个训练过程学习率恒定不变。现已在赋值后追加 `agent_model._setup_lr_schedule()` 重建该闭包，退火才真正生效。详见[训练诊断与修复记录](#12-训练诊断与修复记录)。

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

#### `scripts/rebuild_pool_index.py` — 离线重建对手池索引

对已有的 `checkpoints/agent_gen_*.zip` 逐个重放**修复后**的评估逻辑（`training.evaluate.evaluate_generation`，与训练主循环共用同一份代码），生成一份可信的索引，而不修改/覆盖原始 `pool_index.json`：

```bash
python scripts/rebuild_pool_index.py
# 或指定目录
python scripts/rebuild_pool_index.py --checkpoint-dir checkpoints --out-dir checkpoints
```

输出两个新文件：
- `pool_index_rebuilt.json` — 和训练时产出的结构一致（含容量上限、ELO 淘汰），可直接对照原 `pool_index.json` 观察差异
- `elo_history.csv` — **不受对手池淘汰影响**的完整逐代 ELO/胜率记录，适合画图看"这一轮训练到底学到了多少"

> 用途见[训练诊断与修复记录](#12-训练诊断与修复记录)——已有的 50 个 checkpoint 本身不受评估记账 Bug 影响，可以安全复用，无需重新训练即可拿到一份可信的历史曲线。跑完整 50 代重评估比单代评估慢很多倍，请预留时间。

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
  0.08 ┤╲
       │ ╲
       │  ╲___
  0.03 │      ╲___________
       │                  
  0.01 ┤──────────────────
       0    5   ...   30   ...   50   generation

前30代从0.08线性退火到0.01，之后保持恒定。
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
# 或Windows CMD:
.venv\Scripts\activate.bat

# 或 PowerShell:
.venv\Scripts\Activate.ps1

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

# 运行集成测试（1000局随机游戏）
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

#### 启动 TensorBoard

```bash
# 启动 TensorBoard 查看训练曲线
tensorboard --logdir logs/
# 浏览器打开 http://localhost:6006
```

TensorBoard 会自动发现 `logs/` 目录下所有代（generation）的训练日志。左侧边栏可按 `gen_0`、`gen_1`…… 筛选特定代的曲线。**建议关闭未选中曲线的平滑（smoothing）**——将左侧 `Smoothing` 滑块拖到 0，以便观察真实的波动幅度。

---

#### 训练参数详解

训练过程由三大类超参数共同控制，理解每个参数的含义是读懂曲线的基石。

##### 一、PPO 核心超参数（`training/config.py` → `PPO_CONFIG`）

| 参数 | 默认值 | 含义与作用 |
|------|--------|-----------|
| `n_steps` | 1024 | **每轮采样步数**。20 个并行环境各走 1024 步，共收集 20×1024 = **20,480** 条经验后再做一次 PPO 更新 |
| `batch_size` | 256 | **小批量大小**。20,480 条经验被切分成 20,480/256 = **80 个 mini-batch**，每个 mini-batch 做一次梯度更新 |
| `n_epochs` | 4 | **每轮数据的重复利用次数**。同一批 20,480 条经验，PPO 会反复学习 4 遍（每次重新打乱）。比早期版本的 10 更保守，减少对当前对手过拟合 |
| `learning_rate` | 5e-5 → 1e-5 | **代际余弦退火**（非恒定值！）。`compute_learning_rate(generation)` 按当前代数计算，每代开始时通过 `agent_model._setup_lr_schedule()` 让新值真正生效——同一代内学习率恒定，跨代呈阶梯式下降。详见[训练诊断与修复记录](#12-训练诊断与修复记录) |
| `gamma` | 0.99 | **折扣因子**。衡量未来奖励的重要性： γ=0.99 意味着 100 步后的奖励仍有 0.99¹⁰⁰≈37% 的权重。双人博弈通常设置较高 |
| `gae_lambda` | 0.92 | **GAE 参数**。控制优势估计的偏差-方差权衡：λ→1 低偏差高方差（看得远但噪声大）；λ→0 高偏差低方差（只看一步但稳定） |
| `clip_range` | 0.15 | **PPO 裁剪范围**。新旧策略概率比被限制在 [0.85, 1.15]，防止单次更新步子过大，是 PPO 稳定性的核心 |
| `ent_coef` | 0.01→退火 | **熵系数**（详见下方退火机制）。加在 loss 上的探索奖励权重——鼓励策略保持一定随机性，避免过早锁定次优动作 |
| `vf_coef` | 0.75 | **价值函数损失权重**。控制 Critic（状态价值估计）在总 loss 中的占比 |
| `max_grad_norm` | 0.5 | **梯度裁剪阈值**。所有参数的梯度范数被限制在 0.5 以内，防止梯度爆炸 |

##### 二、Self-Play 训练流程参数（`SELFPLAY_CONFIG`）

| 参数 | 默认值 | 含义与作用 |
|------|--------|-----------|
| `n_envs` | 20 | **并行环境数**。20 个 `SubprocVecEnv` 子进程同时采样，每进程占用约 1 个 CPU 核心 |
| `generations` | 50 | **总代数**。每代训练 500K 步，总计 50×500K = **25M 环境步** |
| `steps_per_generation` | 500,000 | **每代训练步数**。环境步（含对手自动步），约产生 500,000/(20×1024) ≈ **24 次 rollout**，每次 rollout 做 `n_epochs`=4 遍学习 |
| `opponent_pool_size` | 20 | **对手池容量**。保留最近（按 ELO 淘汰后）的 20 个历史模型作为候选对手 |
| `eval_games` | 100 | **完整 ELO 评估局数**。到 `eval_interval_generations` 间隔时，与池中 ELO 最高的 2 个对手各打 `max(20, eval_games//2)` 局来估算 ELO |
| `eval_interval_generations` | 5 | **完整评估间隔**。每隔 5 代跑一次上面的完整评估；其余每代仍会跑一次更便宜的轻量评估（见 `cheap_eval_games`），不会出现完全没有真实信号的代次 |
| `cheap_eval_games` | 15 | **每代轻量评估局数**（match-pairs）。用于产生每代都真实存在的 `win_rate_vs_prev` 和增量 ELO 更新 |
| `random_opponent_prob` | 0.10 | **随机对手采样概率**。10% 的局面对战随机策略，保持多样性，防止策略坍缩 |
| `latest_opponent_prob` | 0.50 | **最新对手采样概率**。50% 的局面对战最新一代模型（水平最接近，产生有效梯度信号） |
| `ent_start / ent_end` | 0.08 → 0.01 | **熵系数退火范围**。前 30 代从 0.08 线性退火到 0.01 |

##### 三、熵系数退火（Entropy Annealing）

这是 Self-Play 训练中**最重要的调度机制**，直接控制探索-利用的平衡：

```
Generation    0    1    2   ...   15   ...  29   30   31  ...   49
ent_coef    0.08 → 0.075 → 0.070 ... 0.045 ... 0.011 → 0.01 → 0.01 (恒定)
```

| 阶段 | 熵系数 | 策略表现 | 目的 |
|------|--------|---------|------|
| **第 0–5 代**（高熵期） | 0.08–0.068 | 策略高度随机，大量尝试各种动作组合 | 广泛探索动作空间，发现"买牌 vs 拿筹码 vs 留牌"的基本价值 |
| **第 6–20 代**（中熵期） | 0.068–0.026 | 策略开始聚焦，有明显偏好 | 在探索中逐步收敛到有效策略，同时保留一定灵活性 |
| **第 21–30 代**（退火尾段） | 0.026–0.01 | 策略接近确定性 | 精细优化胜率，减少噪声动作 |
| **第 31–50 代**（低熵期） | 0.01 | 几乎确定性策略，仅保留极少随机性 | 在稳定的对手分布下追求极致胜率 |

> **健康信号**：熵系数退火过程中 `entropy_loss` 应**平稳下降**而非断崖式暴跌。若高熵期 entropy 就已经很低（<-3.5），说明策略过早坍缩——可增大 `ent_start`。

---

#### TensorBoard 曲线全解

以下逐一解析训练过程中 TensorBoard 可能出现的关键曲线，包括正常形态、异常信号和应对措施。这些曲线在每个 generation 下独立记录（tag 名如 `gen_0/train/approx_kl`）。

> **当前项目实际产出的 tag**（已用真实训练日志核对过）：`time/fps`、`train/approx_kl`、`train/clip_fraction`、`train/clip_range`、`train/entropy_loss`、`train/explained_variance`、`train/learning_rate`、`train/loss`、`train/policy_gradient_loss`、`train/value_loss`、`train/n_updates`。
>
> **下面 ①②两节描述的 `rollout/ep_rew_mean`、`rollout/ep_len_mean` 目前并不会出现**——SB3 只有在环境被 `stable_baselines3.common.monitor.Monitor` 包装时才会记录这两个 tag，而 `training/self_play_env.py`/`self_play_loop.py` 目前没有做这层包装（`SelfPlayEnv` 直接喂给 `SubprocVecEnv`）。这两节保留是因为它们描述的是标准 PPO 训练该有的诊断方法，一旦给环境加上 `Monitor` 包装就能直接对上；在此之前，请改用 `checkpoints/pool_index.json`（[见下方 Self-Play 特有指标](#⑫-self-play-特有指标当前实际存放位置)）里的 `elo` / `win_rate_vs_prev` 来判断训练是否在真正进步。

---

##### ① `rollout/ep_rew_mean` — 平均回合奖励（⚠ 当前项目未记录，见上方说明）

| 属性 | 说明 |
|------|------|
| **含义** | 当前 generation 内所有完成对局的**智能体方平均奖励**（不含对手奖励） |
| **计算方式** | 每局结束时，若智能体获胜→+1.0，失败→-1.0，平局→0，加上过程中累积的塑形奖励（±0.02–0.08/步）。按 SB3 的 100-episode 滑动窗口取均值 |
| **正常形态** | 早期低且波动大（-0.3 ~ +0.2），中期逐步上升（+0.1 ~ +0.4），后期趋于平稳（+0.2 ~ +0.5） |
| **理想趋势** | **整体向上**但允许代际波动——Self-Play 中对手也在变强，偶尔回落 10–20% 是正常现象 |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 始终在 0 附近窄幅振荡（±0.1） | 策略未学到有效行为，或对手太弱/太强导致无梯度信号 | 检查对手采样分布；增大 `ent_coef` 初始值 |
| 某代突然从 +0.4 暴跌至 -0.3 | 新对手策略"克制"了当前智能体的习惯 | 正常——等 2-3 代后观察是否回升，Self-Play 有自愈能力 |
| 连续 5+ 代持续下降 | 策略退化（policy collapse），可能因熵过早消失 | 增大 `ent_start` 到 0.08，减小 `n_epochs` 到 5 |
| 波动幅度极大（±0.5 以上） | 回合奖励方差大，可能因为终局 ±1 信号过于稀疏 | 增大 `n_steps` 到 4096 以收集更多完整对局 |

---

##### ② `rollout/ep_len_mean` — 平均回合长度（⚠ 当前项目未记录，见上方说明）

| 属性 | 说明 |
|------|------|
| **含义** | 每局游戏的平均步数（含双方动作，直至终局条件触发） |
| **正常范围** | 双人 Splendor 通常在 **40–90 步**之间（20–45 回合/人） |
| **趋势解读** | 早期随机策略可能快速结束（乱买牌约 30–40 步）；中期学到"刷筹码→买牌"后对局变长（60–80 步）；后期双方势均力敌时趋于稳定 |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 回合长度持续缩短（< 30 步） | 一方被迅速碾压（实力差距过大），或筹码枯竭死锁 | 检查对手强度；确认跳步动作（#50）未被滥用 |
| 回合长度持续增长（> 100 步） | 双方都保守（只刷筹码不买牌），拖延结束 | 增大买牌塑形奖励，减小囤积惩罚 |
| 长度骤变伴随 reward 骤变 | 策略发生了质变（如学会了"纯买低级卡速攻"） | 正常——观察趋势是否可持续，无需干预 |

---

##### ③ `train/policy_gradient_loss` — 策略梯度损失

| 属性 | 说明 |
|------|------|
| **含义** | PPO-Clipped 目标函数的值。**负值越大**（如 -0.02）表示策略在朝有利方向更新；接近 0 表示更新幅度小 |
| **正常形态** | 持续小幅负值（-0.01 ~ -0.03），偶尔出现正值的尖刺（< 0.005） |
| **核心解读** | 此 loss **不是监督学习中的"越低越好"**——它反映了 PPO 在最大化优势加权策略概率。只要不剧烈震荡，数值大小不代表好坏 |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 长期接近 0（±0.001） | 策略几乎停止更新——梯度消失或 clip 过紧 | 减小 `n_epochs` 或增大 `clip_range` 到 0.25；检查学习率是否太低 |
| 频繁大幅尖刺（> 0.05） | 策略更新步子过大，新旧策略分歧严重 | 减小 `learning_rate` 到 1e-4；增大 `n_steps` 收集更准确的优势估计 |
| 整体趋势性发散（从 -0.01 → -0.1 → ...） | Loss 在恶化 | 立即检查 `approx_kl` 是否飙升——可能需重置学习率 |

---

##### ④ `train/value_loss` — 价值函数损失

| 属性 | 说明 |
|------|------|
| **含义** | Critic 网络对状态价值的预测误差（MSE）。衡量"预判局面好坏"的准确性 |
| **正常形态** | 早期很高（30–80），随着训练稳步下降到 10–20，后期趋于平稳 |
| **关键洞察** | 此 loss 下降说明 Critic 越来越准确地估计每个局面的胜率——这对 GAE 优势计算的精度至关重要 |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 一直很高（> 50）不下降 | Critic 无法学习——可能是 reward 尺度过大（终局 ±1 vs 塑形 ±0.04 差距 25 倍） | 正常——终局信号本身方差大，允许 value_loss 维持较高 |
| 突然跳水到极低（< 1） | Critic 过拟合当前对手，丧失泛化能力 | 正常在换对手时会回升——观察下一代 |
| 与 `ep_rew_mean` 同步波动 | Critic 和策略在同步适应 | **理想状态**——说明 Actor-Critic 协作良好 |

---

##### ⑤ `train/entropy_loss` — 策略熵（探索度）

| 属性 | 说明 |
|------|------|
| **含义** | 当前策略在 51 个动作上的**平均信息熵**。熵高=策略"犹豫"（各动作概率接近均匀）；熵低=策略"果断"（强烈偏好少数动作） |
| **正常范围** | 动作数 51，理论最大熵 = ln(51) ≈ 3.93。正常训练从 ≈ 2.5–3.0 下降到 ≈ 1.5–2.0 |
| **趋势** | 应处于**稳定下降通道**而非断裂式暴跌 |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 前 3 代就已降至 < 1.0（过早坍缩） | 策略过早锁定少数动作，后续探索不足 | 增大 `ent_start` → 0.08 或 0.10；减慢退火速率 |
| 退火完毕后始终 > 2.5（探索过度） | 策略未学到有效区分，停留在随机水平 | 检查 reward 是否合理；增大 `n_epochs`；确认动作掩码工作正常 |
| 每代初期熵高、末期熵低（锯齿状） | 换对手时策略"重新探索"→ 逐步收敛 | **理想模式**——说明 Self-Play 的对手多样性在起作用 |

> **从截图观察**：TensorBoard 中 entropy_loss 曲线呈现典型的**阶梯式下降**——每代开始时因新对手而轻微回升，然后在该代内逐步下降。这是 Self-Play 区别于固定环境训练的标志性特征。

---

##### ⑥ `train/approx_kl` — 近似 KL 散度

| 属性 | 说明 |
|------|------|
| **含义** | 新旧策略之间的**近似 KL 散度**（Kullback-Leibler divergence），衡量每次 PPO 更新后策略"变了多少" |
| **正常范围** | 0.002–0.015 |
| **核心解读** | **这是 PPO 稳定性最重要的监控指标**。KL 散度过高说明策略在一轮更新中变化过大，可能违反 PPO 的信任域原则 |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 稳定在 0.003–0.008 | 策略在 PPO 信任域内平滑更新 | ✅ **理想状态** |
| 频繁超过 0.02 | 策略更新步子太大，`clip_range` 可能限制不住 | 减小 `learning_rate` 到 1e-4；减小 `n_epochs` 到 5；确认 `max_grad_norm`=1.0 |
| 突然飙升（> 0.05） | 策略发生了剧烈变化，可能在某批数据上严重过拟合 | 增大 `batch_size` 到 1024 以平滑梯度；检查是否有异常对局数据 |
| 长期接近 0（< 0.001） | 策略几乎不动——学习停滞 | 检查 `entropy_loss` 是否也接近 0；可能梯度消失 |
| **每代初期出现尖刺 → 迅速回落** | 换对手时策略需要较大幅度调整 → PPO 裁剪介入 → 迅速稳定 | **Self-Play 的正常模式**——截图中的尖刺正是此现象 |

---

##### ⑦ `train/clip_fraction` — PPO 裁剪比例

| 属性 | 说明 |
|------|------|
| **含义** | 在本次 PPO 更新中，被 `clip_range` 裁剪的概率比的**比例**。例如 clip_fraction=0.15 意味着 15% 的动作概率变化被 PPO 强制限制 |
| **正常范围** | 0.05–0.20 |
| **核心解读** | clip_fraction 应该**非零但不高**。零→策略更新太小（PPO 裁剪未触发，可能学习率过低）；过高（>0.3）→大量动作被裁剪，说明新旧策略差异大 |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 0.08–0.18 | PPO 裁剪正常起作用，有约束但不过度 | ✅ **理想状态** |
| 接近 0（< 0.02） | 策略更新太小，学习停滞 | 增大 `learning_rate` 或减小 `n_steps` |
| 持续 > 0.25 | 大量更新被裁剪——策略在信任域边缘反复碰撞 | 与 `approx_kl` 高企配合诊断——减小 `learning_rate` |
| 初期高 → 逐步收敛到 0.05–0.10 | 训练逐渐稳定 | **正常趋势**——截图中的 clip_fraction 正是此模式 |

---

##### ⑧ `train/explained_variance` — 解释方差

| 属性 | 说明 |
|------|------|
| **含义** | Critic 的价值预测能"解释"多少实际回报的方差。范围 [-∞, 1]，**越接近 1 越好** |
| **正常范围** | 0.2–0.7，波动较大 |
| **核心解读** | explained_variance ≈ 1：Critic 完美预测回报（过拟合风险）；≈ 0：不比猜平均值好；< 0：比瞎猜还差（严重问题） |

**诊断指南**：

| 现象 | 诊断 | 建议 |
|------|------|------|
| 稳定在 0.3–0.6 | Critic 能捕捉价值的基本变化 | ✅ **可接受**——Splendor 的随机性和非完美信息限制了上限 |
| 频繁跌入负值区域（< 0） | Critic 预测严重偏离实际——可能是对手变化导致旧估值失效 | Self-Play 换代时常见——通常在几千步内恢复；若持续为负则检查 reward 计算 |
| 逼近 0.9+ | Critic 几乎完美预测——可能过拟合当前对手 | 正常——但需警惕换对手后的剧烈回调 |

> **从截图观察**：explained_variance 曲线波动剧烈且偶尔跌入负值是 Self-Play 的典型特征——每次更换对手后 Critic 需要重新校准。不必惊慌，关注其**移动平均趋势**而非单点数值。

---

##### ⑨ `train/learning_rate` — 学习率

| 属性 | 说明 |
|------|------|
| **含义** | 当前 PPO 优化器的学习率。本项目采用**代际余弦退火**：`compute_learning_rate(generation)` 从 5e-5 退火到 1e-5，跨 50 代 |
| **形态** | **代内恒定、跨代阶梯式下降**的曲线——不是平滑的线性/余弦曲线，因为退火是按"代数"而非按"环境步"计算的（每代开始时算一次目标值，代内保持不变） |
| **历史提醒** | 早期版本这里曾经是**一条水平直线**——退火计算是对的，但赋值 `agent_model.learning_rate = compute_learning_rate(generation)` 之后没有重建 SB3 内部实际读取的 `lr_schedule` 闭包，导致赋值是死代码，全程学习率恒定在初始值。现已在赋值后追加 `agent_model._setup_lr_schedule()` 修复，详见[训练诊断与修复记录](#12-训练诊断与修复记录)。若你看到这条曲线仍是水平直线，说明修复未生效，先检查这一行 |

---

##### ⑩ `train/loss` — 总损失

| 属性 | 说明 |
|------|------|
| **含义** | PPO 的**综合损失** = policy_gradient_loss + `vf_coef`×value_loss - `ent_coef`×entropy_loss |
| **正常形态** | 初始较高（3–8），逐步下降并趋于 0.1–0.5 |
| **注意** | 这个复合 loss 的三个分量含义迥异——policy loss 越小（越负）越好、value loss 越小越好、entropy loss（被减去）越大越好。因此**总 loss 下降不必然意味着训练顺利**，建议优先关注各分量 |

---

##### ⑪ `train/n_updates` — 梯度更新次数

| 属性 | 说明 |
|------|------|
| **含义** | PPO 模型已完成的**梯度更新次数** |
| **计算** | n_updates = 累计环境步数 / (n_envs × n_steps) × n_epochs |
| **形态** | 单调递增的**直线**——仅用于确认训练在持续更新 |
| **每代关系** | 每代 500K 步 → 500,000/(20×1024)×4 ≈ **98 次更新** |

---

##### ⑫ Self-Play 特有指标（当前实际存放位置）

早期文档曾把这些写成 TensorBoard 自定义 tag（`eval/elo` 等），但**当前代码从未把它们写入 TensorBoard**——它们只会打印到训练 stdout，并持久化进 `checkpoints/pool_index.json`（完整字段说明见本节后方「Checkpoint 文件详解 → pool_index.json — 对手池索引」）。要跟踪这些指标，请读取 `pool_index.json` 或用 `python scripts/rebuild_pool_index.py` 重建后再读取，而不是在 TensorBoard 里找：

| 指标（pool_index.json 字段） | 含义 | 健康范围 |
|------|------|---------|
| `elo` | 当前代对抗对手池 ELO 最高的 2 个对手估算出的 ELO 分（每代都有值，但只有 `elo_source="full_eval"` 的代次是完整评估；其余是更粗略的 `incremental_vs_latest`） | 应整体上升（从 1200 起步） |
| `win_rate_vs_prev` | 对抗最新对手的胜率（第0代为 `null`） | 0.50–0.65 为健康区间（微幅进步）；>0.80 可能对手太弱；<0.40 说明落后 |
| `elo_source` / `win_rate_source` | 数据来源标注：`full_eval` / `incremental_vs_latest`（或 `cheap_vs_latest`）/ `baseline` | 用于判断该代数值的置信度——只看 `full_eval` 的代次做长期趋势判断更可靠 |

> 池子本身（`OpponentPool`，容量 20，按 ELO 淘汰）不等于"全部训练历史"——想看不受淘汰影响的完整 50 代曲线，用 `scripts/rebuild_pool_index.py` 生成的 `elo_history.csv`。

---

#### 综合诊断速查表

将上述指标组合起来看，可以快速判断训练状态。**`ep_rew_mean` 列在当前项目中不可用**（见上方 TensorBoard 曲线全解开头的说明），可改用 `pool_index.json` 里 `elo`/`win_rate_vs_prev` 的趋势作为核心进步信号，与下列各列配合判断：

| 场景 | pool_index elo/win_rate 趋势 | entropy_loss | approx_kl | value_loss | explained_var | 诊断与行动 |
|------|------------|-------------|-----------|------------|--------------|------------|
| 🟢 **健康训练** | 整体上升（允许震荡） | 平稳下降 | 0.003–0.008 | 持续下降 | 0.3–0.6 | 一切正常，继续训练 |
| 🟡 **早期探索不足** | 长期停滞在 1200 附近 | < 1.5（前几代） | 接近 0 | 高 | 低 | 增大 `ent_start` → 0.1 以上 |
| 🟡 **学习停滞** | 横向波动 > 10 代 | 稳定但不再降 | < 0.002 | 不再降 | 平稳 | 适度增大 `learning_rate` 起始值，或减小 `clip_range` |
| 🟡 **PPO 更新过大** | 大幅震荡 | 剧烈波动 | > 0.02 | 震荡 | 剧烈波动 | 减小 `learning_rate`，减小 `n_epochs` |
| 🔴 **策略坍缩** | 连续 ≥5 代持续下降 | < 1.0 | 接近 0 | 下降/持平 | 下降 | 增大 `ent_start`，增加 `random_opponent_prob` |
| 🔴 **训练发散** | NaN 或极端值 | 异常 | > 0.1 | NaN | NaN | 立即停止→减小学习率→从最后正常 checkpoint 恢复 |
| 🔵 **Self-Play 换代震荡** | 骤降后 2–3 代内回升 | 轻微回升 | 出现尖刺 | 轻微回升 | 可能短暂变负 | **正常现象**——无需干预，这是适应新对手的过程 |
| ⚫ **记账失真**（历史 Bug，已修复） | `win_rate_vs_prev` 精确等于 0.5、`elo` 呈完美等差数列 | — | — | — | — | 这不是训练问题，是评估被跳过导致的伪造数据。确认 `training/evaluate.py`/`opponent_pool.py` 已应用[本文档记录的修复](#12-训练诊断与修复记录) |

---

#### 从截图中观察实际训练

> ⚠️ 以下内容描述的是历史截图（可能来自曾经启用过 `Monitor` 包装、因而记录了 `ep_rew_mean` 的某个版本）。**当前代码库不会产生 `ep_rew_mean` tag**（见上方说明），如果你现在打开 TensorBoard 找不到对应曲线，这是预期行为，不是回归。以下描述仅作历史参考，实际进度请看 `pool_index.json`。

训练截图展示了本项目的 TensorBoard 实际输出，关键观察：

1. **ep_rew_mean**（顶行）呈明显**上升趋势**，从初始约 -0.1 波动上升至 +0.2~0.4 区间——确认智能体在持续进步。注意曲线中的**周期性凹陷**：每个凹陷对应 Self-Play 换对手时的暂时不适应，随后快速恢复并创新高。

2. **entropy_loss**（中行）呈现典型的**阶梯式退火**——前 10 代从 -2.0 稳步降至 -3.0 左右，之后保持稳定。每次换代时的小幅回升说明策略在面对新对手时重拾探索。

3. **approx_kl**（中行）维持在 0.005 以下的健康区间，偶有尖刺（换对手时策略需更大调整）但迅速回落——PPO 信任域约束有效。

4. **value_loss**（中行）从初始约 40 持续下降至 10–15——Critic 的估值能力稳步提升，这与 ep_rew_mean 的上升是互为因果的。

5. **explained_variance**（底行）整体位于 0.2–0.6 区间，偶有跌落负值——这是 Self-Play 环境下 Critic 的预期表现，无需担忧。

6. **clip_fraction** 从初始约 0.15 逐步收敛到 0.05–0.08——说明训练早期策略快速收敛后被裁剪较多，后期策略趋于稳定。

> **结论**：截图中的训练曲线呈现出**教科书般的健康 Self-Play 训练模式**——所有指标均在其预期范围内波动，没有策略坍缩、发散或过拟合的迹象。

---

### Checkpoint（检查点）文件详解

训练过程中，`checkpoints/` 目录会自动保存模型快照。每次保存产生**两种格式**的同一份模型，以及对手池元数据。

#### 目录结构总览

```
checkpoints/
├── agent_gen_0/              ← 文件夹格式（SB3 原始 save）
│   ├── data                      训练元数据（缓冲区、episode 信息等）
│   ├── policy.pth                策略网络权重（Actor + Critic）~6.3 MB
│   ├── policy.optimizer.pth      优化器状态（Adam 动量）~12.7 MB
│   ├── pytorch_variables.pth     PyTorch 训练变量（lr, clip_range 等）
│   ├── _stable_baselines3_version SB3 版本号
│   └── system_info.txt           训练环境的系统信息
│
├── agent_gen_0.zip            ← ZIP 压缩格式（SB3 标准存档）~19 MB
├── agent_gen_1/  +  .zip      ← 第 1 代
├── ...
├── agent_gen_N/  +  .zip      ← 第 N 代
│
├── agent_latest/              ← 最新代模型的文件夹副本
├── agent_latest.zip           ← 最新代模型的 ZIP 副本
│
└── pool_index.json            ← 对手池索引（ELO、胜率、代数、数据来源标注）
```

---

#### 一、两种存储格式：文件夹 vs ZIP

SB3 的 `model.save(path)` 根据路径后缀决定输出格式：

| 格式 | 产生方式 | 大小 | 用途 |
|------|---------|------|------|
| **文件夹** (`agent_gen_0/`) | `model.save("checkpoints/agent_gen_0")`（无后缀） | ~19 MB | 方便直接查看内部文件、调试 |
| **ZIP 压缩包** (`agent_gen_0.zip`) | `model.save("checkpoints/agent_gen_0.zip")`（含 `.zip`） | ~19 MB | 标准存档格式，便于传输和版本管理 |

> 本项目**两种格式同时保存**（见 `self_play_loop.py`中的 `model.save(checkpoint_path)`）。ZIP 是实际被对手池加载的格式（`MaskablePPO.load(path)` 两种格式均可加载）。

---

#### 二、每个模型内部 6 个文件详解

##### ① `policy.pth` — 策略网络权重（**核心文件**）

| 属性 | 说明 |
|------|------|
| **大小** | ~6.3 MB |
| **内容** | PyTorch `state_dict`，包含 Actor（策略头）和 Critic（价值头）以及 `SplendorFeatureExtractor`（特征提取器）的**全部参数** |
| **包含的参数** | 特征提取器（203→512→512→512→256）+ Actor 头（256→512→512→51）+ Critic 头（256→512→512→1），总计 ~1,045,000 个参数 |
| **查看方式** | `torch.load("policy.pth", map_location="cpu")` → 返回 `OrderedDict`，键名如 `policy.optimizer` 相关参数、`features_extractor.*`、`pi.*`、`vf.*` 等 |

**查看示例**：

```python
import torch

# 读取策略权重
state_dict = torch.load("checkpoints/agent_gen_0/policy.pth", map_location="cpu")

# 查看所有层的名称和形状
for key, tensor in state_dict.items():
    print(f"{key:50s} shape={list(tensor.shape)}")
```

输出示例（部分）：
```
features_extractor.fc1.weight    shape=[512, 203]
features_extractor.fc1.bias      shape=[512]
features_extractor.fc2.weight    shape=[512, 512]
features_extractor.fc3.weight    shape=[512, 512]
features_extractor.fc4.weight    shape=[256, 512]
mlp_extractor.policy_net.0.weight shape=[512, 256]
mlp_extractor.value_net.0.weight shape=[512, 256]
action_net.weight                shape=[51, 512]
value_net.weight                 shape=[1, 512]
```

##### ② `policy.optimizer.pth` — 优化器状态

| 属性 | 说明 |
|------|------|
| **大小** | ~12.7 MB（约为权重的 2×，因为 Adam 为每个参数存储 first-moment 和 second-moment） |
| **内容** | Adam 优化器的 `state_dict`，包含每个参数的 `exp_avg`（一阶矩）、`exp_avg_sq`（二阶矩）、`step` 计数 |
| **关键作用** | **恢复训练时必需**——若只用 `policy.pth` 加载权重而丢失优化器状态，Adam 的动量会从零开始，需要额外几千步重新"预热" |
| **格式** | PyTorch 标准 `optimizer.state_dict()` |

```python
# 查看优化器状态
opt_state = torch.load("checkpoints/agent_gen_0/policy.optimizer.pth", map_location="cpu")
print(f"学习率: {opt_state['param_groups'][0]['lr']}")

# 查看第一个参数的动量统计
param_id = list(opt_state['state'].keys())[0]
print(f"exp_avg shape: {opt_state['state'][param_id]['exp_avg'].shape}")
print(f"exp_avg_sq shape: {opt_state['state'][param_id]['exp_avg_sq'].shape}")
print(f"当前 step: {opt_state['state'][param_id]['step']}")
```

##### ③ `pytorch_variables.pth` — 训练变量

| 属性 | 说明 |
|------|------|
| **大小** | ~1.3 KB |
| **内容** | PPO 算法状态的标量/向量变量，包括当前 `learning_rate`、`clip_range`、`ent_coef`、`n_steps` 累积计数等 |
| **作用** | 确保恢复训练时所有超参数和计数器与保存时完全一致 |

```python
import io, torch
# 从 ZIP 中直接读取（无需解压）
import zipfile
with zipfile.ZipFile("checkpoints/agent_gen_0.zip") as z:
    with z.open("pytorch_variables.pth") as f:
        variables = torch.load(io.BytesIO(f.read()), map_location="cpu")
for k, v in variables.items():
    print(f"{k}: {v}")
```

##### ④ `data` — 训练元数据

| 属性 | 说明 |
|------|------|
| **大小** | ~38 KB |
| **内容** | SB3 内部状态缓冲区：当前的 `RolloutBuffer`（观察、动作、奖励、优势估计等）、episode 统计累计值、随机数种子偏移量 |
| **作用** | 确保 `model.learn()` 恢复时**精确接续**采样和更新循环（避免重复/遗漏步数） |

##### ⑤ `_stable_baselines3_version` — 版本标记

| 属性 | 说明 |
|------|------|
| **大小** | 5 字节 |
| **内容** | 保存时使用的 SB3 版本号（如 `2.3.0`） |
| **作用** | 加载时校验版本兼容性——SB3 会检查此版本号，跨大版本可能无法加载 |

##### ⑥ `system_info.txt` — 系统信息

| 属性 | 说明 |
|------|------|
| **大小** | ~190 字节 |
| **内容** | 训练环境的操作系统、Python 版本、PyTorch 版本、CUDA 可用性等 |
| **作用** | 问题追溯——当加载旧模型出现兼容性问题时，可对照原始环境 |

---

#### 三、`pool_index.json` — 对手池索引

这是一个独立的 JSON 数组，记录对手池中所有历史模型的元信息，是 Self-Play 的"记分牌"：

```json
[
  {
    "path": "D:\\...\\checkpoints\\agent_gen_0.zip",
    "generation": 0,
    "elo": 1200.0,
    "win_rate_vs_prev": null,
    "elo_source": "baseline",
    "win_rate_source": "baseline"
  },
  {
    "path": "D:\\...\\checkpoints\\agent_gen_1.zip",
    "generation": 1,
    "elo": 1198.4,
    "win_rate_vs_prev": 0.4,
    "elo_source": "incremental_vs_latest",
    "win_rate_source": "cheap_vs_latest"
  },
  {
    "path": "D:\\...\\checkpoints\\agent_gen_5.zip",
    "generation": 5,
    "elo": 1203.05,
    "win_rate_vs_prev": 0.5,
    "elo_source": "full_eval",
    "win_rate_source": "full_eval"
  }
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `path` | `str` | 模型 ZIP 文件的绝对路径——被 `MaskablePPO.load()` 加载 |
| `generation` | `int` | 该模型产自第几代训练 |
| `elo` | `float` | 估算的 ELO 评分。是否为完整评估结果由 `elo_source` 标注（而非再靠小数/整数猜测） |
| `win_rate_vs_prev` | `float \| null` | 对抗最新对手的胜率——< 0.5 说明落后，> 0.55 说明显著进步。第 0 代为 `null`（无上一代可比） |
| `elo_source` | `str` | `"full_eval"`（完整评估，到 `eval_interval_generations` 间隔时产生）/ `"incremental_vs_latest"`（每代都跑的轻量评估）/ `"baseline"`（第0代） |
| `win_rate_source` | `str` | `"full_eval"` / `"cheap_vs_latest"` / `"baseline"`，含义同上 |

> **历史提醒**：早期版本没有 `elo_source`/`win_rate_source` 字段，且非评估代直接写死 `win_rate_vs_prev=0.5`、`elo=best_elo+10`——如果你看到旧的 `pool_index.json` 里 `win_rate_vs_prev` 全是精确的 0.5、`elo` 呈现完美等差数列，那就是这个历史 Bug 的产物，不代表真实训练效果。详见[训练诊断与修复记录](#12-训练诊断与修复记录)。已有的 50 个 checkpoint 本身不受影响，可用 `python scripts/rebuild_pool_index.py` 重新生成一份可信的索引。

**采样机制**：训练时，`OpponentPool` 按以下概率从此 JSON 中采样对手：
- 50% → 最新条目（`get_latest_entry()`，按 `generation` 显式选取，与当前策略最接近，产生有效梯度信号）
- 40% → ELO 加权采样（偏好强对手，推动进化）
- 10% → 均匀随机（保持多样性，防止策略坍缩）

**满员处理**：池容量 ≥ 20 时，移除 ELO 最低的条目再添加新模型（不改变其余条目的顺序）。

---

#### 四、`agent_latest` 与 `agent_latest.zip`

| 文件 | 说明 |
|------|------|
| `agent_latest/` | 最新一代模型的**文件夹副本**（与最新的 `agent_gen_N/` 内容相同） |
| `agent_latest.zip` | 最新一代模型的 **ZIP 副本** |

两者在每代训练结束后被 `agent_model.save(latest_path)` 覆盖更新（见 `self_play_loop.py` 保存 checkpoint 的步骤）。作用是提供一个**稳定的入口**——无需知道当前是第几代，始终加载最新的：

```python
from sb3_contrib import MaskablePPO

# 始终加载最新模型，无需知道代次
model = MaskablePPO.load("checkpoints/agent_latest.zip")
```

---

#### 五、如何加载与使用

##### 加载完整模型（含优化器状态，可继续训练）

```python
from sb3_contrib import MaskablePPO

# 从 ZIP 加载（推荐，标准方式）
model = MaskablePPO.load("checkpoints/agent_gen_5.zip")

# 或从文件夹加载
model = MaskablePPO.load("checkpoints/agent_gen_5")

# 恢复训练
model.set_env(new_vec_env)
model.learn(total_timesteps=500_000)
```

##### 仅加载策略权重（推理/对战，不含优化器）

```python
import torch
from sb3_contrib import MaskablePPO

# 方式一：SB3 标准加载后提取
model = MaskablePPO.load("checkpoints/agent_gen_5.zip")
# model.policy 即可用于 predict()

# 方式二：手动加载权重（更轻量）
state_dict = torch.load("checkpoints/agent_gen_5/policy.pth", map_location="cpu")
# 过滤出策略相关的键
policy_weights = {k: v for k, v in state_dict.items() if "optimizer" not in k}
```

##### 直接读取 ZIP 内的文件（无需解压到磁盘）

```python
import io, zipfile, torch

with zipfile.ZipFile("checkpoints/agent_gen_5.zip") as z:
    # 列出所有文件
    for name in z.namelist():
        info = z.getinfo(name)
        print(f"{name:35s} {info.file_size:>10,d} bytes")

    # 读取系统信息
    with z.open("system_info.txt") as f:
        print(f.read().decode())

    # 读取策略权重（无需解压到磁盘）
    with z.open("policy.pth") as f:
        state_dict = torch.load(io.BytesIO(f.read()), map_location="cpu")
```

##### 查看对手池演化历史

```python
import json
with open("checkpoints/pool_index.json") as f:
    pool = json.load(f)

# 打印 ELO 演化轨迹（win_rate_vs_prev 第0代为 None，需要判空）
for entry in sorted(pool, key=lambda e: e["generation"]):
    wr = entry["win_rate_vs_prev"]
    wr_str = f"{wr:.2%}" if wr is not None else "N/A (baseline)"
    print(f"Gen {entry['generation']:2d} | ELO: {entry['elo']:7.1f} "
          f"[{entry['elo_source']:>20s}] | Win vs Prev: {wr_str}")
```

输出示例（数值来自一次真实的 4 代迷你验证跑，而非旧版本那种伪造数据）：
```
Gen  0 | ELO:  1200.0 [            baseline] | Win vs Prev: N/A (baseline)
Gen  1 | ELO:  1198.4 [incremental_vs_latest] | Win vs Prev: 40.00%
Gen  2 | ELO:  1203.0 [            full_eval] | Win vs Prev: 50.00%
Gen  3 | ELO:  1201.4 [incremental_vs_latest] | Win vs Prev: 40.00%
```

##### 恢复训练命令行

```bash
# 从第 10 代 checkpoint 继续训练
python scripts/train.py --resume checkpoints/agent_gen_10.zip
```

---

#### 六、文件大小与磁盘规划

| 组件 | 单个大小 | 50 代总计 |
|------|---------|----------|
| 文件夹格式 (`agent_gen_N/`) | ~19 MB | —（通常只保留最近的） |
| ZIP 格式 (`agent_gen_N.zip`) | ~19 MB | ~950 MB |
| `agent_latest/` + `.zip` | ~38 MB | 恒定 |
| `pool_index.json` | ~2–5 KB | 恒定 |
| **合计（完整 50 代 ZIP）** | | **~1 GB** |

> 磁盘空间充裕（> 100 GB 空闲）时无需清理。若需精简，可以安全删除：
> - `agent_gen_N/` 文件夹（保留 `.zip` 即可，ZIP 内容完全相同）
> - 早期代次的 `agent_gen_0.zip ~ agent_gen_9.zip`（对手池只保留最近 20 个）

---

## 11. 硬件优化说明

### CPU 并行采样

`SubprocVecEnv` 启动 20 个独立 Python 子进程，每个运行一个 `SelfPlayEnv` 实例。24 核 CPU 留出 4 核给系统和其他任务。

### GPU 显存利用

当前配置下 GPU 显存占用约 **50MB**（模型 4MB + 缓冲区 35MB + 梯度 8MB），16GB 中剩余 ~15.95GB。这意味着：

- **可以增大模型**：加宽隐藏层（512→1024）、加深网络
- **可以增大 Batch**：`n_steps` 从 1024 增至 4096+，`batch_size` 从 256 增至 1024+
- **可以增大并行**：`n_envs` 从 20 增至 32（若 CPU 允许）

调整这些参数可参看 `training/config.py`。

### 内存使用

20 个环境进程 × ~20KB/状态 + Python 进程开销 ≈ **< 500MB**，远在 32GB 限制内。

---

## 附录：常见问题

**Q: 训练时 GPU 利用率低？**
A: 增大 `n_envs` 或 `n_steps` 来喂更多数据给 GPU。检查 `training/config.py` 中的参数。

**Q: 智能体只学会刷筹码不买牌？**
A: 增大 `ent_start` 初始值（0.08→0.12 左右），或增加购牌塑形奖励权重。

**Q: 对手池里全是相似的对手？**
A: 增大 `random_opponent_prob`（0.10→0.20），增加随机对手采样概率。

**Q: 某代突然变差？**
A: 正常现象——Self-Play 中对手策略变化会导致暂时震荡，通常几代后恢复。

**Q: `pool_index.json` 里 `win_rate_vs_prev` 一直不变、但 `elo` 却在稳定上升，这正常吗？**
A: **不正常，这是已知历史 Bug 的症状**，不代表训练真的有效或无效。判断依据：如果 `win_rate_vs_prev` 精确等于 `0.5`、`elo` 呈现完美等差数列（例如每条记录都恰好 +10），说明当时的评估代码在大多数代次直接跳过了真实评估、写了写死的占位值，而不是真的在打真实对局。这个问题已经修复（`training/evaluate.py` 的 `evaluate_generation()` 现在每代都会做真实的轻量评估，并且每条 `pool_index.json` 记录都带 `elo_source`/`win_rate_source` 标注数据来源），详见[训练诊断与修复记录](#12-训练诊断与修复记录)。如果你手上还有旧版本训练出的 checkpoint，可以用 `python scripts/rebuild_pool_index.py` 重新算一份可信的历史曲线，不需要重新训练。

---

## 12. 训练诊断与修复记录

这一节记录第一轮 50 代训练之后发现并修复的一批评估/记账相关 Bug——它们**不影响模型权重本身**（已保存的 `agent_gen_*.zip` 均可安全复用），但会导致 `checkpoints/pool_index.json` 里的 ELO / 胜率数据失真，以及学习率调度、终局奖励信号出现问题。如果你在阅读旧版训练日志或旧 `pool_index.json` 时发现数字很奇怪，很可能就是下面这几个原因。

### 现象

第一轮训练完成后，`pool_index.json` 中 `win_rate_vs_prev` 从头到尾几乎没有变化，但 `elo` 却在持续上升——两者看起来矛盾。

### 根因（共 5 处，均已修复）

| # | 问题 | 位置 | 症状 |
|---|------|------|------|
| 1 | **评估被跳过导致数据被写死** | `training/self_play_loop.py` | 早期版本只有 `generation % eval_interval_generations == 0` 的代才跑真实评估，其余代直接 `win_rate_vs_prev=0.5`、`elo=best_elo+10`——`eval_interval_generations=5` 时，50 代里只有 9 代是真实数据 |
| 2 | **ELO 每次评估都重置到 1200** | `training/evaluate.py` `estimate_elo()` | 真实评估出的 ELO 因为总是从 1200 基线开始算，数值明显偏低，和 #1 里那条"伪造+10阶梯"完全不在一个量纲上 |
| 3 | **对手池淘汰机制会打乱列表顺序** | `training/opponent_pool.py` `OpponentPool.add()` | 用 `sort()+pop(0)` 实现淘汰会永久按 ELO 重排 `self.entries`，导致"最新对手"（`entries[-1]`）语义损坏；叠加 #2 的量纲错位，真实评估出的记录反而最容易被误判成"最弱"而被淘汰——这正是为什么最终 `pool_index.json` 里一条真实数据都不剩 |
| 4 | **学习率退火是死代码** | `training/self_play_loop.py` | `compute_learning_rate(generation)` 算出的余弦退火值被正确赋给了 `agent_model.learning_rate`，但 SB3 实际读取学习率用的是初始化时生成的 `lr_schedule` 闭包，赋值不会重建它——导致全程训练学习率恒定在初始值，退火从未生效 |
| 5 | **对手把游戏结束时终局奖励被丢弃** | `training/self_play_env.py` `SelfPlayEnv.step()` / `_auto_play_opponent()` | 对手回合的 reward 一律被丢弃；但"输棋"通常是在对手最后一步棋才判定的，这意味着 agent 很可能很少收到真实的 -1 终局信号，价值函数容易偏乐观 |

### 修复方式

1. `training/evaluate.py` 新增 `evaluate_generation()`：每一代都做一次真实的轻量评估（`cheap_eval_games` 局），到 `eval_interval_generations` 间隔时再额外做一次完整评估覆盖当代结果；`PoolEntry` 新增 `elo_source`/`win_rate_source` 字段标注每个数字的可信度，不再有任何凭空写死的占位值。
2. `estimate_elo()` 新增 `prior_elo` 参数，从 agent 自己上一次的已知 ELO 继续更新，不再每次重置为 1200。
3. `OpponentPool.add()` 改用非破坏性的 `min()+remove()` 做淘汰；新增 `get_latest_entry()`，按 `generation` 字段显式取"最新"，不再依赖列表顺序。
4. `self_play_loop.py` 在重新赋值 `agent_model.learning_rate` 之后追加调用 `agent_model._setup_lr_schedule()`，让 SB3 实际读取到新的学习率。
5. `SelfPlayEnv._auto_play_opponent()` 保留对手每一步的 reward；当对手的某一步结束游戏时，取该 reward 的负值（reward.py 的终局值是零和的：+1/-1/0）作为 agent 收到的终局奖励，而不是用 agent 自己回合前的旧塑形奖励。

### 验证方式

- Bug 2/3/5：各自写了独立脚本，用构造好的数据直接验证函数行为，不依赖真实对局。
- Bug 1/4：跑了一次几代的迷你真实训练，确认 `pool_index.json` 每条记录都带来源标注、不再出现精确 0.5 或完美等差数列，且 `train/learning_rate` 在跨代之间确实在下降。
- 遗留数据处理：已有的 50 个 `agent_gen_*.zip` 本身不受这些 Bug 影响（保存逻辑在评估逻辑之前执行），可以用 `scripts/rebuild_pool_index.py` 重新跑一遍评估、生成一份可信的历史记录，不需要重新训练。**但如果要开始新一轮训练，建议从头跑而不是 `--resume`**——因为学习率退火是按"代数 / 总代数"计算的，`--resume` 到较后的代数会让退火进度直接落在末尾，起不到该有的效果。
