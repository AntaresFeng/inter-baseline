"""
中心化贪心分配基线 (Centralized Greedy Assignment Baseline)
——————————————————————————————————————————————————————————
本脚本实现了 simple_spread_v3 环境的一个强启发式策略，用于评估和对比学习算法。

策略描述:
  每一步都采用集中式的全局视角:
  1. 穷举所有智能体到地标的配对排列 (permutations)，选择使总欧氏距离最小的分配方案。
  2. 每个智能体根据分配到的地标方向，使用离散动作 (上/下/左/右/不动) 独立地向目标移动。

定位与用途:
  - 并非理论上的最优回报上限。
    其损失来源: 离散动作无法精确逼近、缺乏多智能体协同避碰、贪心分配不考虑长期收益。
  - 应视为 "Oracle 分配 + 独立反应式导航" 的强基线 (strong baseline)。
    它代表了当“谁去哪个地标”被完美解决，但动作执行仍为朴素离散控制时的性能水平。
  - 若学习算法的回报接近或超过此基线，说明其不仅学会了分配，还可能学到了更优的协同移动或避碰策略。

评估指标:
  - mean_return: 每 episode 总回报 (所有智能体每步奖励均值的累计) 的均值
  - collision_step_rate: 出现碰撞的步数占总步数的比例
  - late_assignment_dist: 每 episode 最后 7 步中最小分配总距离的平均值 (反映末期接近程度)

依赖: pettingzoo[mpe]>=1.24.0
"""

from itertools import permutations
import numpy as np
from pettingzoo.mpe import simple_spread_v3

A_NO = 0
A_LEFT = 1
A_RIGHT = 2
A_DOWN = 3
A_UP = 4


def action_to_target(agent_pos, target_pos, deadband=0.03):
    delta = target_pos - agent_pos
    if abs(delta[0]) >= abs(delta[1]):
        if delta[0] > deadband:
            return A_RIGHT
        if delta[0] < -deadband:
            return A_LEFT
    else:
        if delta[1] > deadband:
            return A_UP
        if delta[1] < -deadband:
            return A_DOWN
    return A_NO


def eval_heuristic(episodes=300):
    returns = []
    collisions = 0
    steps = 0
    nearest_late = []
    for ep in range(episodes):
        env = simple_spread_v3.parallel_env(
            N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False
        )
        obs, _ = env.reset(seed=3000 + ep)
        ep_ret = 0.0
        while env.agents:
            world = env.unwrapped.world
            agents = world.agents
            landmarks = world.landmarks
            apos = np.array([a.state.p_pos for a in agents])
            lpos = np.array([l.state.p_pos for l in landmarks])
            best_perm = None
            best_cost = 1e9
            for perm in permutations(range(3)):
                cost = sum(np.linalg.norm(apos[i] - lpos[perm[i]]) for i in range(3))
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm
            acts = {
                env.agents[i]: action_to_target(apos[i], lpos[best_perm[i]])
                for i in range(3)
            }
            obs, rewards, terms, truncs, infos = env.step(acts)
            ep_ret += float(np.mean([rewards[a] for a in rewards]))
            # collision before/after roughly after step
            col = 0
            for i, a in enumerate(agents):
                for b in agents[i + 1 :]:
                    if np.linalg.norm(a.state.p_pos - b.state.p_pos) < a.size + b.size:
                        col += 1
            collisions += int(col > 0)
            steps += 1
            if steps % 25 >= 18:
                nearest_late.append(best_cost)
        env.close()
        returns.append(ep_ret)
    arr = np.array(returns)
    print(
        {
            "mean_return": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "collision_step_rate": round(collisions / steps, 4),
            "late_assignment_dist": round(float(np.mean(nearest_late)), 4),
        }
    )


eval_heuristic()
