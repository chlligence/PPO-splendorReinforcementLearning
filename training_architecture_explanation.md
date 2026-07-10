# Splendor 强化学习 — 训练架构深度解析

---

## 目录

1. [术语表](#1-术语表)
2. [单代训练流程（微观）](#2-单代训练流程微观)
3. [总体自对弈训练流程（宏观）](#3-总体自对弈训练流程宏观)
4. [PPO 算法深度解析](#4-ppo-算法深度解析)
5. [Stable-Baselines3 框架](#5-stable-baselines3-框架)
6. [SB3-Contrib 与 MaskablePPO](#6-sb3-contrib-与-maskableppo)
7. [神经网络架构](#7-神经网络架构)
8. [对手池 OpponentPool](#8-对手池-opponentpool)
9. [Baseline 基线训练](#9-baseline-基线训练)
10. [Checkpoint 检查点机制](#10-checkpoint-检查点机制)
11. [奖励函数设计（新版）](#11-奖励函数设计新版)

---

## 1. 术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| **智能体** | Agent | 正在被训练的 AI 玩家 |
| **对手** | Opponent | 从对手池中采样出的历史版本 AI，与智能体对弈 |
| **环境** | Environment / Env | Splendor 游戏规则的代码实现 |
| **观察** | Observation / obs | 203 维向量，描述当前棋盘状态 |
| **动作** | Action | 51 种离散操作之一（取宝石、买牌、留牌、跳过） |
| **动作掩码** | Action Mask | 长度 51 的布尔数组，标记哪些动作当前合法 |
| **奖励** | Reward | 单步反馈信号，终局 ±1 为主导 |
| **回合** | Step | 一个玩家执行一次动作 |
| **对局** | Episode / Game | 从开局到终局的完整一局游戏 |
| **策略** | Policy / π | 神经网络，输入观察 → 输出各动作的概率分布 |
| **价值函数** | Value Function / V | 神经网络，输入观察 → 输出该状态的预期胜率 |
| ** rollout** | Rollout | 智能体与环境交互 N 步，收集 (obs, action, reward, mask) 数据 |
| **世代** | Generation | 自对弈的一轮：采样对手 → 训练 N 步 → 加入对手池 |
| **对手池** | Opponent Pool | 保存历史模型检查点的集合（最多 20 个） |
| **检查点** | Checkpoint | 保存在磁盘上的模型参数文件（`.zip`） |
| **ELO** | ELO Rating | 棋类评分系统，用于量化模型相对实力 |
| **PPO** | Proximal Policy Optimization | 近端策略优化算法 |
| **GAE** | Generalized Advantage Estimation | 广义优势估计，平衡偏差-方差的优势函数计算方法 |
| **Entropy** | 策略熵 | 策略输出概率分布的混乱度，高熵 = 多探索，低熵 = 确定性强 |

---

## 2. 单代训练流程（微观）

下面是**一代（One Generation）** 训练中，数据如何从游戏环境流向神经网络并完成一次参数更新的完整过程。

### 2.1 流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    单代训练流程 (Generation N)                            │
│                    500,000 steps ≈ 约 244 局完整游戏                       │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  1. 采样对手          │
    │  OpponentPool.sample()│
    │  → 对手模型检查点路径  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  2. 创建 20 个并行环境 (SubprocVecEnv)                            │
    │                                                                  │
    │  ┌──────────┐  ┌──────────┐       ┌──────────┐                  │
    │  │ Env #0   │  │ Env #1   │  ...  │ Env #19  │                  │
    │  │ P0:Agent │  │ P0:Agent │       │ P0:Agent │                  │
    │  │ P1:Oppnt │  │ P1:Oppnt │       │ P1:Oppnt │                  │
    │  └────┬─────┘  └────┬─────┘       └────┬─────┘                  │
    │       │             │                  │                         │
    │       └─────────────┴──────────────────┘                         │
    │                     │                                            │
    │            SubprocVecEnv (多进程并行)                               │
    └─────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  3. 数据收集循环 (Rollout)                                        │
    │                                                                  │
    │  for step in 1..2048:    ← 每个环境收集 2048 步                    │
    │                                                                  │
    │    ┌─────────────────────────────────────────────┐               │
    │    │ 智能体回合 (Agent Turn)                      │               │
    │    │                                             │               │
    │    │  ① env 返回 obs[203] + action_mask[51]      │               │
    │    │  ② Agent 神经网络前向传播:                    │               │
    │    │     obs → FeatureExtractor(256)              │               │
    │    │         → Actor head → action_logits[51]     │               │
    │    │         → Critic head → state_value[1]       │               │
    │    │  ③ 用 action_mask 屏蔽非法动作 (设为 -∞)       │               │
    │    │  ④ softmax → 概率分布 → 采样 action           │               │
    │    │  ⑤ env.step(action) → 执行动作                │               │
    │    │  ⑥ 【存储】 (obs, action, reward, value,      │               │
    │    │              log_prob, mask)                  │               │
    │    └─────────────────────────────────────────────┘               │
    │                                                                  │
    │    ┌─────────────────────────────────────────────┐               │
    │    │ 对手回合 (Opponent Turn) — 自动消耗            │               │
    │    │                                             │               │
    │    │  ① SelfPlayEnv 检测轮到对手                   │               │
    │    │  ② 对手模型 predict(obs, action_masks=mask)   │               │
    │    │  ③ env.step(opp_action)                      │               │
    │    │  ④ 【丢弃 — 不存入缓冲区】                     │               │
    │    └─────────────────────────────────────────────┘               │
    │                                                                  │
    │  每局结束 → env.reset() → 新游戏开始                              │
    └─────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  4. 缓冲区大小                                                     │
    │                                                                  │
    │  总迁移数 = 2048 steps × 20 envs = 40,960 条                      │
    │  每条 = obs(203×4B) + action(8B) + reward(4B) + value(4B)        │
    │        + log_prob(4B) + mask(51B) ≈ 880 bytes                    │
    │  总 GPU 内存 ≈ 40,960 × 880B ≈ 36 MB                             │
    └─────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  5. PPO 更新循环 (10 epochs)                                      │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────┐        │
    │  │ Step A: GAE 计算优势函数 A_t                             │        │
    │  │                                                       │        │
    │  │ A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...         │        │
    │  │ δ_t = r_t + γ·V(s_{t+1}) - V(s_t)    (TD error)       │        │
    │  │                                                       │        │
    │  │ Returns = A_t + V(s_t)   (用于训练 Critic)             │        │
    │  └──────────────────────────────────────────────────────┘        │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────┐        │
    │  │ for epoch in 1..10:                                     │        │
    │  │   for minibatch (size=512) from 40,960 buffer:          │        │
    │  │                                                       │        │
    │  │     ① 重新前向传播 → new_log_probs, new_values         │        │
    │  │     ② 计算概率比率 r_t = exp(new_log_prob - old_log_prob) │     │
    │  │     ③ 计算 PPO-Clip Loss:                              │        │
    │  │        L_CLIP = min(r_t·A_t, clip(r_t,1-ε,1+ε)·A_t)  │        │
    │  │     ④ 计算 Value Loss:                                 │        │
    │  │        L_VF = (V_predicted - Returns)²                 │        │
    │  │     ⑤ 计算 Entropy Loss:                               │        │
    │  │        L_ENT = -Σ π(a)log π(a)    (鼓励探索)           │        │
    │  │     ⑥ 总 Loss = -L_CLIP + 0.5·L_VF - 0.01·L_ENT      │        │
    │  │     ⑦ 反向传播 → 梯度裁剪 → 更新参数                    │        │
    │  └──────────────────────────────────────────────────────┘        │
    └─────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────┐
    │  6. 保存 Checkpoint   │
    │  → agent_gen_N.zip   │
    │  → agent_latest.zip  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────────────────┐
    │  7. ELO 评估 (每 2 代一次)                 │
    │                                          │
    │  新模型 vs 对手池 Top-3 各 50 局           │
    │  计算 ELO 分数                            │
    │  记录 win_rate_vs_prev                    │
    └──────────┬───────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  8. 加入对手池         │
    │  pool.add(PoolEntry(  │
    │    path=...,          │
    │    generation=N,      │
    │    elo=...,           │
    │    win_rate=...       │
    │  ))                  │
    └──────────────────────┘
```

### 2.2 数据流总结

```
Obs[203] ──→ FeatureExtractor ──→ 256-dim embedding
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
              Actor Head                          Critic Head
          256→512→ReLU                           256→512→ReLU
              │                                      │
              ▼                                      ▼
          512→51 (logits)                      512→1 (value)
              │
              ▼
    Mask(illegal → -∞) → Softmax → Sample action
```

---

## 3. 总体自对弈训练流程（宏观）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    总训练流程 — Self-Play 50 代                           │
│                                                                         │
│  目标: 通过不断与"过去的自己"对弈，实现策略的螺旋式进化                       │
└─────────────────────────────────────────────────────────────────────────┘

                              START
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Generation 0         │
                    │  对手: Random (纯随机) │
                    │                       │
                    │  Agent ──vs── Random   │
                    │  (随机权重)   (随机动作) │
                    │                       │
                    │  训练 500K steps       │
                    │  学会: 合法动作、基础买牌 │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  加入对手池            │
                    │  Pool[0]: gen_0       │
                    │  ELO: ~1200           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Generation 1         │
                    │  对手: gen_0 (弱)      │
                    │                       │
                    │  Agent ──vs── gen_0   │
                    │  (新权重)   (旧权重)    │
                    │                       │
                    │  训练 500K steps       │
                    │  学会: 击败初代自己      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  加入对手池            │
                    │  Pool[0,1]: gen_0,1   │
                    └──────────┬───────────┘
                               │
                               ▼
                              ...
                               │
                               ▼
                    ┌──────────────────────────────────────────────┐
                    │  Generation N (e.g., 25)                      │
                    │  采样对手:                                     │
                    │    50% → gen_{N-1} (最新，水平最接近)          │
                    │    40% → ELO-加权 (偏好强对手)                  │
                    │    10% → 随机抽取 (保持多样性)                  │
                    │                                              │
                    │  Agent ──vs── 采样的历史对手                    │
                    │                                              │
                    │  训练 500K steps                               │
                    │  学会: 击败多样化的历史策略                      │
                    └──────────┬───────────────────────────────────┘
                               │
                               ▼
                              ...
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Generation 50        │
                    │  (最终模型)            │
                    │                       │
                    │  对手池已积累 20 个      │
                    │  从弱到强的历史版本      │
                    │                       │
                    │  最终模型应能击败        │
                    │  所有早期版本            │
                    └──────────────────────┘
                               │
                               ▼
                              END
```

### 3.1 对手池的演变示意

```
Generation:  0     5     10    15    20    25    30    35    40    45    50
             │     │     │     │     │     │     │     │     │     │     │
Pool[0]:  ───█─────█─────█─────█─────█───── (弱，可能被淘汰)
Pool[5]:         ─────█─────█─────█─────█─────█─────
Pool[10]:              ─────█─────█─────█─────█─────█─────█─────
Pool[15]:                    ─────█─────█─────█─────█─────█─────█─────
Pool[20]:                          ─────█─────█─────█─────█─────█─────█─────
...                                                                    ─────█  ← 最强

                    ELO 分数随世代逐步上升 ─────────────→
```

---

## 4. PPO 算法深度解析

### 4.1 PPO 在项目中的角色

PPO（Proximal Policy Optimization）是本项目的**核心学习算法**。它负责：
- 读取游戏过程中收集的 (observation, action, reward) 数据
- 计算"这个动作好不好" → 优势函数 A(s,a)
- 更新神经网络参数，使"好动作"的概率增加，"坏动作"的概率减少
- **约束更新幅度**，防止一次更新太大导致策略崩溃

### 4.2 为什么选择 PPO

| 特性 | PPO | 其他算法 (DQN/REINFORCE/A3C) |
|------|-----|------------------------------|
| 样本效率 | 中等 (On-policy, 每批数据只用一次) | DQN 更高 (Off-policy, 经验回放) |
| 稳定性 | ★★★★★ 最好 | REINFORCE 方差极大 |
| 超参数敏感度 | 低 (默认参数通常有效) | A3C 较高 |
| 离散动作空间 | 天然支持 | 都支持 |
| 动作掩码兼容 | ★★★★★ (MaskablePPO) | 需要手动修改 |
| 实现复杂度 | 中等 | DQN 简单, TRPO 复杂 |

### 4.3 PPO-Clip 核心公式

```
概率比率:  r_t(θ) = π_new(a_t|s_t) / π_old(a_t|s_t)

裁剪目标:
  L_CLIP(θ) = E[ min( r_t·A_t , clip(r_t, 1-ε, 1+ε)·A_t ) ]

  其中 ε = 0.2 (裁剪范围)，A_t = 优势函数

直观理解:
  ┌─ A_t > 0 (好动作，想增加概率):
  │   允许 r_t 最大增长到 1+ε = 1.2
  │   超过 1.2 的部分梯度为 0，不再鼓励
  │
  └─ A_t < 0 (坏动作，想降低概率):
      允许 r_t 最小降低到 1-ε = 0.8
      低于 0.8 的部分梯度为 0，不再惩罚
```

### 4.4 GAE (广义优势估计)

GAE 是本项目中使用的优势函数计算方法，平衡**偏差**和**方差**：

```
单步 TD 误差:  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

GAE 优势:     A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

  其中:  γ = 0.99  (折扣因子 — 关注长期收益)
        λ = 0.95  (GAE 参数 — 偏差-方差权衡)
```

- **λ → 0**：只看单步 TD 误差，**低方差但高偏差**
- **λ → 1**：看全部 Monte Carlo 回报，**无偏但高方差**
- **λ = 0.95**：折中，实践中最优

### 4.5 PPO 在本项目中的超参数

| 参数 | 值 | 作用 |
|------|-----|------|
| `n_steps` | 2048 | 每次 rollout 每个环境收集的步数 |
| `batch_size` | 512 | 每次梯度更新用的小批量大小 |
| `n_epochs` | 10 | 同一批数据重复训练 10 轮 |
| `learning_rate` | 3e-4 | Adam 优化器学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `clip_range` | 0.2 | PPO 裁剪范围 ε |
| `ent_coef` | 0.05→0.005 | 熵系数（退火） |
| `vf_coef` | 0.5 | 价值函数损失权重 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |

---

## 5. Stable-Baselines3 框架

### 5.1 什么是 Stable-Baselines3

Stable-Baselines3 (SB3) 是 **PyTorch 编写的强化学习算法库**，提供：
- 开箱即用的 PPO、A2C、DQN、SAC 等算法实现
- 统一的 `model.learn()` 接口
- 内置的 `SubprocVecEnv` / `DummyVecEnv` 并行环境
- TensorBoard 日志集成
- 模型保存/加载 (`model.save()` / `model.load()`)

### 5.2 SB3 在本项目中的调用链

```python
# 1. 创建模型
from sb3_contrib import MaskablePPO

model = MaskablePPO(
    "MlpPolicy",           # 使用 MLP 策略网络
    vec_env,               # 向量化环境 (20 并行)
    policy_kwargs={        # 自定义网络结构
        "features_extractor_class": SplendorFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [512, 512], "vf": [512, 512]},
    },
    n_steps=2048,          # ← 以上所有超参数
    batch_size=512,
    ...
)

# 2. 训练 — SB3 内部自动完成:
#    rollout 收集 → GAE 计算 → PPO 更新 → 重复
model.learn(total_timesteps=500_000)

# 3. 保存
model.save("checkpoints/agent_gen_5.zip")

# 4. 推理
action, _ = model.predict(obs, action_masks=mask)
```

### 5.3 SB3 的 Rollout + Update 内部机制

```
model.learn(total_timesteps=500_000):

  while total_steps < 500_000:
    │
    ├─ Rollout Phase (收集数据):
    │   for step in 1..n_steps:
    │     action, value, log_prob = model.policy(obs, mask)
    │     obs, reward, done, info = env.step(action)
    │     rollout_buffer.add(obs, action, reward, value, log_prob, mask)
    │
    ├─ GAE Computation (计算优势):
    │   last_value = model.policy(obs) if not done else 0
    │   rollout_buffer.compute_returns_and_advantage(last_value)
    │
    └─ Training Phase (更新参数):
        for epoch in 1..n_epochs:
          for batch in rollout_buffer.sample(batch_size):
            loss = compute_ppo_loss(batch)
            loss.backward()
            optimizer.step()
```

---

## 6. SB3-Contrib 与 MaskablePPO

### 6.1 为什么需要 sb3-contrib

标准的 SB3 `PPO` **不支持**动作掩码（Action Masking）。`sb3-contrib` 是一个社区贡献的扩展包，提供了 `MaskablePPO`。

### 6.2 动作掩码的工作原理

```
标准 PPO (无掩码):
  神经网络 → logits[51] → softmax → 概率分布 → 采样
                                   ↑
                            非法动作也有非零概率！
                            → 大量无效探索

MaskablePPO (有掩码):
  神经网络 → logits[51] → mask: legal[51] → 非法动作设为 -∞
                        → softmax → 概率分布 → 采样
                                   ↑
                            非法动作概率精确为 0
                            → 100% 有效探索
```

### 6.3 掩码的传递路径

```
环境 env.step(action) 返回:
  info = {"action_mask": np.array([True, False, True, ...], dtype=bool)}

     ↓ SB3 内部自动提取

模型 predict(obs, action_masks=mask):
  distribution = policy.get_distribution(obs, action_masks=mask)
  # distribution 已经将非法动作概率置零
  action = distribution.sample()
```

### 6.4 MaskablePPO 与标准 PPO 的差异

| 特性 | PPO | MaskablePPO |
|------|-----|-------------|
| 动作掩码 | ❌ 不支持 | ✅ 原生支持 |
| 掩码传递 | N/A | 通过 `info["action_mask"]` |
| 前向传播 | 标准 logits | logits + mask → valid_logits |
| 损失计算 | 标准 PPO loss | 与标准 PPO 相同 (掩码仅影响采样) |
| 保存/加载 | `model.save()` / `model.load()` | 完全相同 |
| 兼容性 | 所有 SB3 环境 | 需要环境在 info 中返回 "action_mask" |

---

## 7. 神经网络架构

### 7.1 网络全景图

```
                        Obs[203]
                           │
    ┌──────────────────────┴──────────────────────┐
    │        SplendorFeatureExtractor              │
    │                                              │
    │  Linear(203, 512) + LayerNorm + ReLU         │
    │  Linear(512, 512) + LayerNorm + ReLU         │  约 760K 参数
    │  Linear(512, 512) + LayerNorm + ReLU         │
    │  Linear(512, 256) + ReLU                     │
    │                                              │
    │  输出: embedding[256]                         │
    └──────────────┬───────────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
┌──────────────┐          ┌──────────────┐
│  Actor Head  │          │  Critic Head │
│  (策略网络)   │          │  (价值网络)   │
│              │          │              │
│  Linear      │          │  Linear      │
│  (256→512)   │          │  (256→512)   │
│  ReLU        │          │  ReLU        │
│              │          │              │
│  Linear      │          │  Linear      │
│  (512→51)    │          │  (512→1)     │
│              │          │              │
│  输出:       │          │  输出:        │
│  logits[51]  │          │  value[1]    │
│  (动作偏好)   │          │  (状态估值)   │
└──────────────┘          └──────────────┘
```

### 7.2 两个头的分工

| | Actor (策略网络) | Critic (价值网络) |
|------|---------------------|---------------------|
| **输入** | embedding[256] | embedding[256] |
| **输出** | logits[51] — 每个动作的"原始偏好分" | scalar — 当前状态的"预估胜率" |
| **用途** | 选择执行哪个动作 | 计算优势函数 A(s,a) = 实际回报 - V(s) |
| **损失函数** | PPO-Clip Loss | MSE Loss: (V_pred - Returns)² |
| **如果去掉** | 智能体无法决策 | 优势函数只能用原始回报，方差极大 |

### 7.3 参数量统计

| 模块 | 计算 | 参数量 |
|------|------|--------|
| Feature Extractor | 203×512 + 512×512 + 512×512 + 512×256 | ~760K |
| Actor Head | 256×512 + 512×51 | ~154K |
| Critic Head | 256×512 + 512×1 | ~131K |
| **总计** | | **~1M** |

> 约 100 万参数，FP32 下约 4MB GPU 显存，非常轻量。

---

## 8. 对手池 OpponentPool

### 8.1 为什么需要对手池

如果只和随机对手训练（Baseline 模式），智能体只能学会"打败随机操作"。这和人类打牌只和不会玩的人打一样——永远学不到高级策略。

**对手池让智能体与"历史上的自己"对弈**，随着训练的进行，对手越来越强，智能体被迫不断进步。

### 8.2 PoolEntry 数据结构

```python
@dataclass
class PoolEntry:
    path: str              # 检查点文件路径
    generation: int        # 哪个世代产生的
    elo: float             # 与该模型相关的 ELO 评分
    win_rate_vs_prev: float # 加入池时 vs 上一代强者的胜率
```

**各字段的详细含义**：

| 字段 | 含义 | 示例值 | 说明 |
|------|------|--------|------|
| `path` | 模型权重文件的**磁盘路径** | `checkpoints/agent_gen_5.zip` | 用于加载对手模型做推理 |
| `generation` | 该模型在**第几代**训练出来的 | `5` | 数字越大通常越新，但不一定更强 |
| `elo` | 估计的 **ELO 实力分** | `1350.0` | 通过与历史强者对弈计算，类似国际象棋评分 |
| `win_rate_vs_prev` | 加入池时**对抗上一代最强者**的胜率 | `0.62` (62%) | > 0.5 说明新模型更强，< 0.5 说明可能退步 |

### 8.3 对手采样策略

```
从对手池选对手的概率分配：

  50% → 最新对手 (latest)
        └─ 理由: 水平和当前 Agent 最接近，提供最有意义的对抗

  40% → ELO-加权采样 (ELO softmax)
        └─ 理由: 多和强者打，提升更快

  10% → 均匀随机 (uniform)
        └─ 理由: 防止只和一种类型打导致策略退化（多样性）
```

### 8.4 对手池的容量管理

```
池上限: 20 个条目

当池满时:
  → 按 ELO 排序
  → 移除 ELO 最低的条目（最弱的对手）
  → 保留 ELO 最高的 19 个 + 新条目

为什么移除最弱的？
  → 太弱的对手无法提供有效的训练信号
  → 保留强者确保 Agent 始终面对足够挑战
```

### 8.5 对手池的可视化演变

```
Generation 0:     Pool = [gen_0 (ELO: 1200)]
Generation 5:     Pool = [gen_0, gen_1, gen_2, gen_3, gen_4, gen_5]
                           ↑ 旧                            新 ↑
Generation 20:    Pool = [gen_5 (ELO: 1180), ..., gen_20 (ELO: 1450)]
                           ↑ 最弱将被淘汰                最强 ↑
Generation 25:    Pool = [gen_10, gen_12, gen_15, ..., gen_25]
                          (gen_5 因 ELO 最低已被移除)

Generation 50:    Pool = [最强 20 个历史版本，按 ELO 排序]
```

---

## 9. Baseline 基线训练

### 9.1 什么是 Baseline

Baseline 训练是**最简单的训练模式**：智能体只和**完全随机的对手**对弈。

```bash
python scripts/train.py --baseline
```

### 9.2 Baseline vs Self-Play 对比

| | Baseline (基线) | Self-Play (自对弈) |
|------|------------------|------------------------|
| **对手** | 纯随机 | 历史模型池 |
| **对手强度** | 不变，始终很弱 | 随训练逐步增强 |
| **训练效果** | 学会合法动作 + 基础策略 | 学会复杂战术 + 应对多样化对手 |
| **上限** | 低 — 只会打新手 | 高 — 理论可以无限进化 |
| **计算开销** | 低 (单进程即可) | 高 (20 并行环境 + 对手加载) |
| **用途** | 快速验证环境正确性、调试网络结构 | 正式训练，追求最强智能体 |
| **启动命令** | `python scripts/train.py --baseline` | `python scripts/train.py` |

### 9.3 何时使用 Baseline

- 首次运行时验证代码无 bug
- 测试新奖励函数是否合理（用 Baseline 快速看到效果）
- 调试神经网络架构
- 环境刚开发完，还不确定 Self-Play 是否能收敛

---

## 10. Checkpoint 检查点机制

### 10.1 什么是 Checkpoint

Checkpoint 是训练过程中**保存到磁盘的模型参数快照**。它包含了神经网络的**所有权重和偏置值**，可以随时加载恢复。

```python
# 保存
model.save("checkpoints/agent_gen_5.zip")

# 加载
model = MaskablePPO.load("checkpoints/agent_gen_5.zip")
```

### 10.2 本项目的 Checkpoint 策略

```
checkpoints/
├── agent_gen_0.zip       ← 第 0 代训练完毕（加入对手池）
├── agent_gen_1.zip       ← 第 1 代训练完毕
├── agent_gen_2.zip
├── ...
├── agent_gen_49.zip      ← 第 49 代训练完毕
├── agent_latest.zip      ← 始终指向最新一代（快捷路径）
├── baseline_vs_random.zip ← Baseline 模式输出
└── pool_index.json       ← 对手池元数据（非模型文件）
```

### 10.3 Checkpoint 的生命周期

```
训练中:
  agent_gen_N.zip  → 作为对手加载，供后代智能体对弈

评估时:
  agent_gen_N.zip  → 加载两个不同代模型，打 200 局评估 ELO

恢复训练:
  agent_gen_10.zip → 从中断处继续训练 (--resume)

部署时:
  agent_latest.zip → 最终模型，可以和人或其他 AI 对弈
```

### 10.4 恢复训练

```bash
# 假如训练在第 10 代中断了
python scripts/train.py --resume checkpoints/agent_gen_10.zip
```

内部逻辑：
1. 从文件名提取 generation=10
2. 加载模型权重
3. 加载对手池 JSON（恢复历史对手记录）
4. 从 generation=11 继续训练

---

## 11. 奖励函数设计（新版）

用户已更新 `splendor/reward.py` 为**基于势能函数的奖励塑形**。以下是新版设计的解析。

### 11.1 核心思想：战略势能函数

新版奖励不再是简单的"买牌+0.05"，而是构建了一个**综合战略势能函数** `get_strategic_potential()`，评估玩家当前资产对胜利的综合贡献：

```python
strategic_potential =
    ① 基础分 (points × 1.0)              ← 终极目标，权重最高
  + ② 基建分 (total_bonuses × 0.25)       ← 每张已购卡提供永久折扣
  + ③ 深挖奖励 (max_single_bonus × 0.08)  ← 单色 bonus ≥3 时触发
  + ④ 对齐奖励 (alignment × 0.15)         ← 资产颜色 vs 版面高分牌需求
  + ⑤ 筹码分 (tokens × 0.02)             ← 购买力储备
  + ⑥ 黄金分 (gold × 0.025)              ← 万能筹码高弹性
  + ⑦ 手牌分 (reserve 数量)              ← 预留卡牌的策略价值
```

**最终奖励 = 0.04 × (势能差值变化)**：

```python
prev_margin = potential(self) - potential(opponent)  # 动作前
curr_margin = potential(self) - potential(opponent)  # 动作后
reward = 0.04 * (curr_margin - prev_margin)          # 差值变化 × 缩放
```

这种设计的好处：
- **零和性质**：一方势能增加 = 另一方相对势能减少，保持对抗性
- **密集信号**：每一步都有反馈，不必等到终局才知道好坏
- **防止刷分**：势能差值比绝对值更重要

### 11.2 版面需求动态分析

```python
# 扫描 Tier 2/3 高分牌，计算各颜色需求度
demand_weights = Σ (card.cost[color] × card.points)  for each high-point face-up card
```

这使智能体**动态感知**当前版面"什么颜色最重要"，引导其购买与版面需求对齐的卡牌。

### 11.3 行为惩罚（防呆）

```python
# 满筹时拿宝石（非买牌）→ 会触发强制弃牌，属于节奏浪费
if not is_buy and total_tokens >= 10:
    reward -= 0.04
```

### 11.4 与旧版奖励的对比

| 维度 | 旧版 | 新版 |
|------|------|------|
| 奖励密度 | 稀疏（只在买牌/终局有信号） | 密集（每步势能变化） |
| 战略深度 | 仅奖励"买牌"行为 | 奖励"基建布局+颜色对齐+深挖" |
| 版面感知 | 无 | 动态分析高分牌颜色需求 |
| 零和性质 | 仅终局 ±1 | 势能差值维护零和趋势 |
| 防止刷分 | 依赖小系数 | 势能差值 + 行为惩罚 |

---
