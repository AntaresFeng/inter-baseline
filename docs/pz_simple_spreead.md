# Simple Spread 环境文档
Simple Spread 是 PettingZoo 中多智能体粒子环境（MPE）的子环境，核心目标是让多个智能体在覆盖所有地标时避免相互碰撞。

## 重要迁移提示
环境 `pettingzoo.mpe.simple_spread_v3` 已迁移至新的 [MPE2 包](https://mpe2.farama.org/)，PettingZoo 未来版本将移除该环境，建议将导入路径更新为 `mpe2.simple_spread_v3`。

## 环境核心规格
| 项目 | 说明 |
|------|------|
| 智能体数量 | 默认 3 个（可通过参数 `N` 调整），智能体标识为 `agent_0, agent_1, agent_2...` |
| 动作空间类型 | 离散（默认）/ 连续 |
| 动作空间形状 | (5) |
| 动作值范围 | 离散：`Discrete(5)`（对应无动作、左移、右移、下移、上移）；连续：`Box(0.0, 1.0, (5))` |
| 观察空间形状 | (18) |
| 观察值范围 | (-∞, +∞) |
| 观察内容 | `[自身速度, 自身位置, 地标相对位置, 其他智能体相对位置, 通信信息]` |
| 状态空间形状 | (54,) |
| 状态值范围 | (-∞, +∞) |
| Parallel API 支持 | 是 |
| 手动控制支持 | 否 |

## 奖励机制
智能体的总奖励由**全局奖励**和**局部惩罚**加权组成，权重由 `local_ratio` 参数控制：
- 全局奖励：基于所有智能体到每个地标最近距离的总和计算（距离越短，奖励越高），权重为 `1 - local_ratio`；
- 局部惩罚：智能体与其他智能体发生碰撞时，每次碰撞惩罚 -1，权重为 `local_ratio`。

## 环境参数
初始化环境时可配置以下参数：
```python
simple_spread_v3.env(
    N=3,            # 智能体和地标的数量（默认3）
    local_ratio=0.5,# 局部奖励权重（全局奖励权重=1-local_ratio，默认0.5）
    max_cycles=25,  # 游戏终止前的帧数（每个智能体行动一步算一帧，默认25）
    continuous_actions=False, # 动作空间是否为连续型（默认False，离散型）
    dynamic_rescaling=False   # 是否根据屏幕尺寸调整智能体/地标大小（默认False）
)
```

## 使用示例
### 1. AEC 模式（异步执行）
```python
from pettingzoo.mpe import simple_spread_v3

# 初始化环境
env = simple_spread_v3.env(render_mode="human")
env.reset(seed=42)

# 智能体迭代执行
for agent in env.agent_iter():
    # 获取当前智能体的最新状态
    observation, reward, termination, truncation, info = env.last()

    # 终止/截断时动作设为None
    if termination or truncation:
        action = None
    else:
        # 示例：随机采样动作（实际场景替换为自定义策略）
        action = env.action_space(agent).sample()
    
    # 执行动作
    env.step(action)

# 关闭环境
env.close()
```

### 2. Parallel 模式（并行执行）
```python
from pettingzoo.mpe import simple_spread_v3

# 初始化并行环境
env = simple_spread_v3.parallel_env(render_mode="human")
observations, infos = env.reset(seed=42)

# 多智能体并行交互
while env.agents:
    # 示例：为所有智能体随机采样动作（实际场景替换为自定义策略）
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # 执行动作并获取新状态
    observations, rewards, terminations, truncations, infos = env.step(actions)

# 关闭环境
env.close()
```

## 核心 API
### raw_env 函数
基础环境初始化函数，可自定义所有核心参数：
```python
pettingzoo.mpe.simple_spread.simple_spread.raw_env(
    N=3,
    local_ratio=0.5,
    max_cycles=25,
    continuous_actions=False,
    render_mode=None,
    dynamic_rescaling=False
)
```
该函数返回未封装的原始环境，适用于需要自定义环境封装逻辑的场景。