"""
MAPPO adapter for highway-env's multi-agent intersection environment.

Fixes:
  1. target_speeds missing in MultiAgentAction's action_config
     (falls back to MDPVehicle.DEFAULT_TARGET_SPEEDS [20,25,30] instead of [0,4.5,9])
  2. observation_config loses detailed Kinematics parameters from IntersectionEnv

Adapts interface for mappo.py:
  - get_obs_size(), get_state_size(), get_action_size()
  - get_avail_actions(), get_state(), n_agents
  - reset()/step() return stacked per-agent observations
  - step() accepts vector actions and returns shared scalar reward
  - arrival reward is emitted once per agent, then that agent is idled
"""

import numpy as np
from highway_env.envs.intersection_env import (
    IntersectionEnv,
    MultiAgentIntersectionEnv,
)

from env.config import deep_merge


def _stack_obs(obs) -> np.ndarray:
    """Convert highway-env's tuple observations to MAPPO's agent batch."""
    if isinstance(obs, tuple):
        obs = np.stack([np.asarray(agent_obs).reshape(-1) for agent_obs in obs])
    else:
        obs = np.asarray(obs)
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
    return obs.astype(np.float32, copy=False)


def _shared_reward(reward) -> float:
    """MAPPO uses one critic target, so per-agent rewards are averaged."""
    arr = np.asarray(reward, dtype=np.float32)
    return float(arr.mean())


class HighwayWrapper(MultiAgentIntersectionEnv):
    """MultiAgentIntersectionEnv with corrected config and MAPPO-compatible interface.

    Instead of hand-patching individual fields, this class rebuilds the config
    by deep-merging the full inheritance chain:
        AbstractEnv → IntersectionEnv → MultiAgentIntersectionEnv
    then overlays any user-provided overrides via a patched ``configure()``.
    """

    def __init__(
        self,
        map_name: str = "intersection-multi-agent-v1",
        agent_ids: bool = True,
        **kwargs,
    ) -> None:
        if map_name != "intersection-multi-agent-v1":
            raise ValueError(
                "HighwayWrapper only supports map_name='intersection-multi-agent-v1'"
            )

        self.map_name = map_name
        self.agent_ids = agent_ids
        self.agents_arrived = np.zeros(0, dtype=bool)
        self.agents_newly_arrived = np.zeros(0, dtype=bool)
        self.agents_active = np.zeros(0, dtype=bool)
        render_mode = kwargs.pop("render_mode", None)
        super().__init__(config=kwargs or None, render_mode=render_mode)

    @staticmethod
    def _inject_nested(config: dict) -> None:
        """Move merged outer-level fields into the inner nested dicts.

        highway-env's multi-agent wrappers (MultiAgentAction,
        MultiAgentObservation) read config from a nested dict
        (action_config / observation_config), but single-agent envs put
        the same fields at the outer level.  After deep-merge the source
        fields land on the outer dict; this moves them inward.
        """
        source = IntersectionEnv.default_config()
        for outer_key, inner_key in [
            ("action", "action_config"),
            ("observation", "observation_config"),
        ]:
            outer = config.get(outer_key, {})
            inner = outer.get(inner_key, {})
            for key in source.get(outer_key, {}):
                if key != "type" and key in outer and key not in inner:
                    inner[key] = outer.pop(key)
            outer[inner_key] = inner

    @classmethod
    def default_config(cls) -> dict:
        # Deep-merge the full inheritance chain so nested dicts (action,
        # observation, …) are preserved at every level instead of replaced.
        base = super(IntersectionEnv, IntersectionEnv).default_config()
        intersection = IntersectionEnv.default_config()
        multi_agent = MultiAgentIntersectionEnv.default_config()
        config = deep_merge(base, intersection, multi_agent)
        cls._inject_nested(config)
        return config

    def configure(self, config: dict | None) -> None:
        """Deep-merge user overrides, then re-home nested keys.

        Overridden from AbstractEnv to use deep_merge instead of shallow
        dict.update(), and to re-inject nested keys so user-provided
        observation/action params reach the inner dicts where
        MultiAgentAction / MultiAgentObservation read them.
        """
        if config:
            self.config = deep_merge(self.config, config)
            self._inject_nested(self.config)

    @property
    def n_agents(self) -> int:
        return len(self.controlled_vehicles)

    def get_obs_size(self) -> int:
        """Single agent observation dimension (flattened)."""
        shape = self.observation_type.agents_observation_types[0].space().shape
        return int(np.prod(shape))

    def get_state_size(self) -> int:
        """Global state = concatenation of all agents' observations."""
        return self.get_obs_size() * self.n_agents

    def get_action_size(self) -> int:
        """Single agent action dimension."""
        return int(self.action_type.agents_action_types[0].space().n)

    @staticmethod
    def _idle_action(agent_action_type) -> int:
        return int(agent_action_type.actions_indexes.get("IDLE", 1))

    def _current_arrived(self) -> np.ndarray:
        return np.array(
            [self.has_arrived(vehicle) for vehicle in self.controlled_vehicles],
            dtype=bool,
        )

    def _refresh_agent_status(self) -> None:
        self.agents_arrived = self._current_arrived()
        self.agents_newly_arrived = np.zeros(self.n_agents, dtype=bool)
        self.agents_active = ~self.agents_arrived

    def _ensure_agent_status(self) -> None:
        if self.agents_arrived.shape != (self.n_agents,):
            self._refresh_agent_status()

    def _coerce_actions(self, action) -> tuple[int, ...]:
        if isinstance(action, np.ndarray):
            action = action.reshape(-1).tolist()
        elif not isinstance(action, (tuple, list)):
            action = [action]
        actions = tuple(int(agent_action) for agent_action in action)
        if len(actions) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} actions, received {len(actions)}"
            )
        return actions

    def _idle_arrived_actions(self, action: tuple[int, ...]) -> tuple[int, ...]:
        self._ensure_agent_status()
        actions = list(action)
        for i, has_arrived in enumerate(self.agents_arrived):
            if has_arrived:
                actions[i] = self._idle_action(self.action_type.agents_action_types[i])
        return tuple(actions)

    def _event_rewards(self, previous_arrived: np.ndarray) -> tuple[float, ...]:
        rewards = []
        for i, vehicle in enumerate(self.controlled_vehicles):
            if previous_arrived[i]:
                reward = 0.0
            else:
                reward = IntersectionEnv._agent_reward(self, 0, vehicle)
            rewards.append(float(reward))
        return tuple(rewards)

    def _update_agent_status(self, previous_arrived: np.ndarray) -> None:
        now_arrived = self._current_arrived()
        self.agents_newly_arrived = np.logical_and(~previous_arrived, now_arrived)
        self.agents_arrived = np.logical_or(previous_arrived, now_arrived)
        crashed = np.array(
            [vehicle.crashed for vehicle in self.controlled_vehicles], dtype=bool
        )
        self.agents_active = ~(self.agents_arrived | crashed)

    def _augment_info(
        self,
        info: dict,
        agents_rewards: tuple[float, ...],
    ) -> dict:
        agents_terminated = tuple(
            bool(arrived or vehicle.crashed)
            for arrived, vehicle in zip(self.agents_arrived, self.controlled_vehicles)
        )
        info["agents_rewards"] = agents_rewards
        info["agents_terminated"] = agents_terminated
        info["agents_arrived"] = tuple(bool(v) for v in self.agents_arrived)
        info["agents_newly_arrived"] = tuple(
            bool(v) for v in self.agents_newly_arrived
        )
        info["agents_active"] = tuple(bool(v) for v in self.agents_active)
        return info

    def get_avail_actions(self) -> np.ndarray:
        """(n_agents, action_size) boolean mask of available actions."""
        self._ensure_agent_status()
        agent_types = self.action_type.agents_action_types
        n_actions = int(agent_types[0].space().n)
        avail = np.zeros((self.n_agents, n_actions), dtype=bool)
        for i, agent_action_type in enumerate(agent_types):
            if self.agents_arrived[i]:
                avail[i, self._idle_action(agent_action_type)] = True
            else:
                for a in agent_action_type.get_available_actions():
                    avail[i, a] = True
        return avail

    def get_state(self) -> np.ndarray:
        """Global state: flatten concatenation of all agents' observations."""
        obs_list = self.observation_type.observe()  # tuple of per-agent obs
        return np.concatenate([np.asarray(o).reshape(-1) for o in obs_list]).astype(
            np.float32, copy=False
        )

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._refresh_agent_status()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            zero_rewards = tuple(0.0 for _ in range(self.n_agents))
            info = self._augment_info(info, zero_rewards)
            return _stack_obs(obs), info
        return _stack_obs(result)

    def step(self, action):
        self._ensure_agent_status()
        previous_arrived = self.agents_arrived.copy()
        action = self._idle_arrived_actions(self._coerce_actions(action))
        obs, _reward, _terminated, truncated, info = super().step(action)
        self._update_agent_status(previous_arrived)
        agents_rewards = self._event_rewards(previous_arrived)
        crash_terminal = any(
            bool(vehicle.crashed and not was_arrived)
            for vehicle, was_arrived in zip(self.controlled_vehicles, previous_arrived)
        )
        terminated = bool(crash_terminal or self.agents_arrived.all())
        info = self._augment_info(info, agents_rewards)
        return (
            _stack_obs(obs),
            _shared_reward(agents_rewards),
            terminated,
            bool(truncated),
            info,
        )
