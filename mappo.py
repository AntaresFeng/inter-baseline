import torch
import tyro
import copy
import datetime
import json
import math
import os
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from pathlib import Path
import shutil
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    env_type: str = "highway"
    """ highway, Pettingzoo, SMAClite ... """
    env_name: str = "intersection-multi-agent-v1"
    """ Name of the environment"""
    env_family: str = "mpe"
    """ Env family when using pz"""
    env_config_path: str | None = None
    """ Path to a UTF-8 JSON file with environment config overrides"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 8
    """ Number of episodes to collect in each rollout"""
    actor_model: str = "mlp"
    """Actor model type: mlp or attention"""
    critic_model: str = "mlp"
    """Critic model type: mlp or attention"""
    actor_hidden_dim: int = 128
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 2
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 2
    """ Number of hidden layers of critic network"""
    attention_feature_dim: int = 7
    """Fallback features per entity when the environment does not expose shape metadata"""
    attention_embed_dim: int = 64
    """Entity embedding dimension for attention models"""
    attention_heads: int = 2
    """Number of attention heads"""
    attention_dropout: float = 0.0
    """Dropout probability applied to attention weights"""
    attention_presence_feature_idx: int = 0
    """Feature index used as the entity presence mask"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD(lambda) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = False
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    epochs: int = 3
    """ Number of training epochs"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    log_every: int = 10
    """ Logging steps """
    clip_gradients: float = -1
    """ No gradient clipping when <= 0; clip at this max norm when > 0"""
    eval_steps: int = 10
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    save_checkpoints: bool = True
    """ Save actor policy checkpoints during and after training"""
    checkpoint_dir: str = "runs"
    """ Root directory for TensorBoard events and policy checkpoints"""
    checkpoint_interval: int = 0
    """ Save a checkpoint every N evaluations; disabled when 0"""
    checkpoint_best_metric: str = "eval/ep_reward"
    """ Evaluation metric used to select the best checkpoint"""
    checkpoint_best_mode: str = "max"
    """ Best checkpoint mode: max or min"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""

    def __post_init__(self):
        for name in ["actor_model", "critic_model"]:
            value = getattr(self, name)
            if value not in {"mlp", "attention"}:
                raise ValueError(f"{name} must be 'mlp' or 'attention', got {value!r}")
        if self.checkpoint_best_mode not in {"max", "min"}:
            raise ValueError(
                "checkpoint_best_mode must be 'max' or 'min', "
                f"got {self.checkpoint_best_mode!r}"
            )
        if self.checkpoint_interval < 0:
            raise ValueError(
                f"checkpoint_interval must be >= 0, got {self.checkpoint_interval}"
            )
        if self.attention_feature_dim <= 0:
            raise ValueError(
                f"attention_feature_dim must be > 0, got {self.attention_feature_dim}"
            )
        if self.attention_embed_dim <= 0:
            raise ValueError(
                f"attention_embed_dim must be > 0, got {self.attention_embed_dim}"
            )
        if self.attention_heads <= 0:
            raise ValueError(f"attention_heads must be > 0, got {self.attention_heads}")
        if self.attention_embed_dim % self.attention_heads != 0:
            raise ValueError(
                "attention_embed_dim must be divisible by attention_heads, "
                f"got {self.attention_embed_dim} and {self.attention_heads}"
            )
        if not (0 <= self.attention_dropout < 1):
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {self.attention_dropout}"
            )
        if self.attention_presence_feature_idx < 0:
            raise ValueError(
                "attention_presence_feature_idx must be >= 0, "
                f"got {self.attention_presence_feature_idx}"
            )


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        state_space,
        action_space,
        normalize_reward=False,
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.obs_space)
        ).to(self.device)
        avail_actions = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        actions = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        log_probs = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        reward = torch.zeros((self.buffer_size, max_length)).to(self.device)
        states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        next_states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        terminated = torch.zeros((self.buffer_size, max_length)).to(self.device)
        truncated = torch.zeros((self.buffer_size, max_length)).to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(
            self.device
        )
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            avail_actions[i, :length] = self.episodes[i]["avail_actions"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            next_states[i, :length] = self.episodes[i]["next_states"]
            terminated[i, :length] = self.episodes[i]["terminated"]
            truncated[i, :length] = self.episodes[i]["truncated"]
            mask[i, :length] = 1
        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)
        self.episodes = [None] * self.buffer_size
        return (
            obs.float(),
            actions.long(),
            log_probs.float(),
            reward.float(),
            states.float(),
            next_states.float(),
            avail_actions.bool(),
            terminated.float(),
            truncated.float(),
            mask,
        )


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x, avail_action=None, deterministic=False):
        logits = self.logits(x, avail_action)
        distribution = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = distribution.sample()
        return action, distribution.log_prob(action)

    def logits(self, x, avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EntityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SelfAttention(nn.Module):
    def __init__(self, feature_size, heads, dropout_factor=0.0) -> None:
        super().__init__()
        if feature_size % heads != 0:
            raise ValueError(
                f"feature_size must be divisible by heads, got {feature_size}/{heads}"
            )
        self.feature_size = feature_size
        self.heads = heads
        self.features_per_head = feature_size // heads
        self.value_all = nn.Linear(feature_size, feature_size, bias=False)
        self.key_all = nn.Linear(feature_size, feature_size, bias=False)
        self.query_all = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_combine = nn.Linear(feature_size, feature_size, bias=False)
        self.dropout = nn.Dropout(dropout_factor)

    def _project(self, layer, x):
        batch_size, n_entities, _ = x.shape
        return (
            layer(x)
            .view(batch_size, n_entities, self.heads, self.features_per_head)
            .permute(0, 2, 1, 3)
        )

    def forward(self, ego, others, mask=None):
        batch_size = ego.shape[0]
        input_all = torch.cat((ego, others), dim=1)
        n_entities = input_all.shape[1]

        key_all = self._project(self.key_all, input_all)
        value_all = self._project(self.value_all, input_all)
        query_all = self._project(self.query_all, input_all)

        attention_mask = None
        if mask is not None:
            attention_mask = mask.view(batch_size, 1, 1, n_entities)

        value, attention_matrix = attention(
            query_all, key_all, value_all, attention_mask, self.dropout
        )
        value = (
            value.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, n_entities, self.feature_size)
        )
        result = (self.attention_combine(value) + input_all) / 2
        return result, attention_matrix


class EgoAttention(nn.Module):
    def __init__(self, feature_size, heads, dropout_factor=0.0) -> None:
        super().__init__()
        if feature_size % heads != 0:
            raise ValueError(
                f"feature_size must be divisible by heads, got {feature_size}/{heads}"
            )
        self.feature_size = feature_size
        self.heads = heads
        self.features_per_head = feature_size // heads
        self.value_all = nn.Linear(feature_size, feature_size, bias=False)
        self.key_all = nn.Linear(feature_size, feature_size, bias=False)
        self.query_ego = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_combine = nn.Linear(feature_size, feature_size, bias=False)
        self.dropout = nn.Dropout(dropout_factor)

    def _project_all(self, layer, x):
        batch_size, n_entities, _ = x.shape
        return (
            layer(x)
            .view(batch_size, n_entities, self.heads, self.features_per_head)
            .permute(0, 2, 1, 3)
        )

    def forward(self, ego, others, mask=None):
        batch_size = ego.shape[0]
        input_all = torch.cat((ego, others), dim=1)
        n_entities = input_all.shape[1]

        key_all = self._project_all(self.key_all, input_all)
        value_all = self._project_all(self.value_all, input_all)
        query_ego = (
            self.query_ego(ego)
            .view(batch_size, 1, self.heads, self.features_per_head)
            .permute(0, 2, 1, 3)
        )

        attention_mask = None
        if mask is not None:
            attention_mask = mask.view(batch_size, 1, 1, n_entities)

        value, attention_matrix = attention(
            query_ego, key_all, value_all, attention_mask, self.dropout
        )
        value = (
            value.permute(0, 2, 1, 3).contiguous().view(batch_size, self.feature_size)
        )
        result = (self.attention_combine(value) + ego.squeeze(1)) / 2
        return result, attention_matrix


def validate_entity_shape(input_dim, entity_shape, presence_feature_idx, label):
    if len(entity_shape) != 2:
        raise ValueError(f"{label} entity_shape must have 2 values, got {entity_shape}")
    entities, features = (int(entity_shape[0]), int(entity_shape[1]))
    if entities <= 0 or features <= 0:
        raise ValueError(f"{label} entity_shape values must be > 0, got {entity_shape}")
    if entities * features != input_dim:
        raise ValueError(
            f"{label} entity_shape {entity_shape} does not match input_dim {input_dim}"
        )
    if presence_feature_idx >= features:
        raise ValueError(
            f"attention_presence_feature_idx={presence_feature_idx} is out of "
            f"range for {features} features"
        )
    return entities, features


class AttentionActor(nn.Module):
    def __init__(
        self,
        input_dim,
        entity_shape,
        output_dim,
        embed_dim,
        heads,
        dropout_factor,
        presence_feature_idx,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.entity_count, self.feature_dim = validate_entity_shape(
            input_dim, entity_shape, presence_feature_idx, "actor"
        )
        self.presence_feature_idx = presence_feature_idx

        self.ego_embedding = EntityMLP(self.feature_dim, embed_dim)
        self.others_embedding = EntityMLP(self.feature_dim, embed_dim)
        self.self_attention_layer = SelfAttention(embed_dim, heads, dropout_factor)
        self.attention_layer = EgoAttention(embed_dim, heads, dropout_factor)
        self.output_layer = EntityMLP(embed_dim, embed_dim, output_dim)

    def _reshape_entities(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected actor input last dimension {self.input_dim}, "
                f"got {x.shape[-1]}"
            )
        leading_shape = x.shape[:-1]
        entities = x.reshape(-1, self.entity_count, self.feature_dim)
        return entities, leading_shape

    def _split_input(self, entities):
        ego = entities[:, 0:1, :]
        others = entities[:, 1:, :]
        mask = entities[:, :, self.presence_feature_idx] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        entities, leading_shape = self._reshape_entities(x)
        ego, others, mask = self._split_input(entities)
        ego = self.ego_embedding(ego)
        others = self.others_embedding(others)
        self_att, _ = self.self_attention_layer(ego, others, mask)
        ego = self_att[:, 0:1, :]
        others = self_att[:, 1:, :]
        ego_attention, attention_matrix = self.attention_layer(ego, others, mask)
        return ego_attention, attention_matrix, leading_shape

    def act(self, x, avail_action=None, deterministic=False):
        logits = self.logits(x, avail_action)
        distribution = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = distribution.sample()
        return action, distribution.log_prob(action)

    def logits(self, x, avail_action=None):
        ego_attention, _, leading_shape = self.forward_attention(x)
        logits = self.output_layer(ego_attention).reshape(
            *leading_shape, self.output_dim
        )
        if avail_action is not None:
            logits = logits.masked_fill(~avail_action, -1e9)
        return logits

    def get_attention_matrix(self, x):
        _, attention_matrix, leading_shape = self.forward_attention(x)
        return attention_matrix.reshape(*leading_shape, *attention_matrix.shape[1:])


class AttentionCritic(nn.Module):
    def __init__(
        self,
        input_dim,
        entity_shape,
        embed_dim,
        heads,
        dropout_factor,
        presence_feature_idx,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.entity_count, self.feature_dim = validate_entity_shape(
            input_dim, entity_shape, presence_feature_idx, "critic"
        )
        self.presence_feature_idx = presence_feature_idx

        self.embedding = EntityMLP(self.feature_dim, embed_dim)
        self.self_attention_layer = SelfAttention(embed_dim, heads, dropout_factor)
        self.output_layer = EntityMLP(embed_dim, embed_dim, 1)

    def _reshape_entities(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected critic input last dimension {self.input_dim}, "
                f"got {x.shape[-1]}"
            )
        leading_shape = x.shape[:-1]
        entities = x.reshape(-1, self.entity_count, self.feature_dim)
        return entities, leading_shape

    def forward(self, x):
        entities, leading_shape = self._reshape_entities(x)
        mask = entities[:, :, self.presence_feature_idx] < 0.5
        embedded = self.embedding(entities)
        empty_others = embedded[:, 1:, :]
        self_att, _ = self.self_attention_layer(embedded[:, 0:1, :], empty_others, mask)

        valid = (~mask).to(self_att.dtype).unsqueeze(-1)
        pooled = (self_att * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        value = self.output_layer(pooled)
        return value.reshape(*leading_shape, 1)


def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    attention_probs = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_probs = dropout(attention_probs)
    return torch.matmul(attention_probs, value), attention_probs


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "highway":
        from env.highway_wrapper import HighwayWrapper

        env = HighwayWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    # elif env_type == "smaclite":
    #     from env.smaclite_wrapper import SMACliteWrapper

    #     env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    # elif env_type == "lbf":
    #     from env.lbf import LBFWrapper

    #     env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "pz":
        from env.pettingzoo_wrapper import PettingZooWrapper

        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")

    return env


def infer_entity_shape(input_dim, feature_dim, label):
    if input_dim % feature_dim != 0:
        raise ValueError(
            f"Cannot infer {label} entity shape: input_dim {input_dim} is not "
            f"divisible by attention_feature_dim {feature_dim}"
        )
    return input_dim // feature_dim, feature_dim


def resolve_entity_shape(
    env,
    method_name,
    input_dim,
    fallback_feature_dim,
    presence_feature_idx,
    label,
):
    method = getattr(env, method_name, None)
    if method is not None:
        entity_shape = method()
    else:
        entity_shape = infer_entity_shape(input_dim, fallback_feature_dim, label)
    return validate_entity_shape(
        input_dim=input_dim,
        entity_shape=entity_shape,
        presence_feature_idx=presence_feature_idx,
        label=label,
    )


def optional_env_value(env, method_name):
    method = getattr(env, method_name, None)
    if method is None:
        return None
    return method()


def make_actor(args, env):
    if args.actor_model == "mlp":
        return Actor(
            input_dim=env.get_obs_size(),
            hidden_dim=args.actor_hidden_dim,
            num_layer=args.actor_num_layers,
            output_dim=env.get_action_size(),
        )
    obs_entity_shape = resolve_entity_shape(
        env=env,
        method_name="get_obs_entity_shape",
        input_dim=env.get_obs_size(),
        fallback_feature_dim=args.attention_feature_dim,
        presence_feature_idx=args.attention_presence_feature_idx,
        label="actor",
    )
    return AttentionActor(
        input_dim=env.get_obs_size(),
        entity_shape=obs_entity_shape,
        output_dim=env.get_action_size(),
        embed_dim=args.attention_embed_dim,
        heads=args.attention_heads,
        dropout_factor=args.attention_dropout,
        presence_feature_idx=args.attention_presence_feature_idx,
    )


def make_critic(args, env):
    if args.critic_model == "mlp":
        return Critic(
            input_dim=env.get_state_size(),
            hidden_dim=args.critic_hidden_dim,
            num_layer=args.critic_num_layers,
        )
    state_entity_shape = resolve_entity_shape(
        env=env,
        method_name="get_state_entity_shape",
        input_dim=env.get_state_size(),
        fallback_feature_dim=args.attention_feature_dim,
        presence_feature_idx=args.attention_presence_feature_idx,
        label="critic",
    )
    return AttentionCritic(
        input_dim=env.get_state_size(),
        entity_shape=state_entity_shape,
        embed_dim=args.attention_embed_dim,
        heads=args.attention_heads,
        dropout_factor=args.attention_dropout,
        presence_feature_idx=args.attention_presence_feature_idx,
    )


def norm_d(grads, d):
    valid_grads = [g.detach() for g in grads if g is not None]
    if not valid_grads:
        return torch.tensor(0.0)
    norms = torch.stack([torch.linalg.vector_norm(g, d) for g in valid_grads])
    total_norm_d = torch.linalg.vector_norm(norms, d)
    return total_norm_d


def reset_env(env, seed=None):
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def scalar_flag(flag):
    arr = np.asarray(flag)
    if arr.shape == ():
        return bool(arr)
    return bool(arr.all())


def shared_scalar_reward(reward):
    arr = np.asarray(reward, dtype=np.float32)
    if arr.shape == ():
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    raise ValueError(
        "This MAPPO implementation expects a shared scalar reward. "
        "Per-agent rewards require changing the critic target and buffer shapes."
    )


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return repr(value)


def format_json(value):
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=json_default,
    )


def load_env_config(path):
    if path is None:
        return {}

    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "env_config_path must be valid JSON: "
            f"{config_path} (line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Failed to read env_config_path {config_path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            "env_config_path must contain a JSON object at the top level: "
            f"{config_path}"
        )
    return config


def markdown_table_value(value):
    if isinstance(value, (dict, list, tuple, set)):
        text = format_json(value)
    elif isinstance(value, (np.ndarray, np.generic, os.PathLike)):
        text = str(json_default(value))
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def save_actor_checkpoint(
    path,
    actor,
    args,
    env_config,
    step,
    training_step,
    num_episodes,
    eval_index,
    checkpoint_type,
    eval_metrics,
    best_metric_value,
):
    payload = {
        "actor_state_dict": {
            name: value.detach().cpu() for name, value in actor.state_dict().items()
        },
        "args": vars(args).copy(),
        "env_config": copy.deepcopy(env_config),
        "step": step,
        "training_step": training_step,
        "num_episodes": num_episodes,
        "eval_index": eval_index,
        "checkpoint_type": checkpoint_type,
        "eval_metrics": None if eval_metrics is None else dict(eval_metrics),
        "best_metric_value": best_metric_value,
    }
    torch.save(payload, path)


if __name__ == "__main__":
    args = tyro.cli(Args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ## import the environment
    kwargs = load_env_config(args.env_config_path)
    env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=copy.deepcopy(kwargs),
    )
    eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=copy.deepcopy(kwargs),
    )

    ## Initialize the actor, critic and target-critic networks
    actor = make_actor(args, env).to(device)
    critic = make_critic(args, env).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    run_dir = Path(args.checkpoint_dir) / f"MAPPO-{run_name}"
    if args.save_checkpoints:
        run_dir.mkdir(parents=True, exist_ok=True)
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MAPPO-{run_name}",
        )
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    env_summary = [
        ("env_type", args.env_type),
        ("env_name", args.env_name),
        ("env_family", args.env_family),
        ("env_config_path", args.env_config_path),
        ("agent_ids", args.agent_ids),
        ("env_class", type(env).__name__),
        ("n_agents", env.n_agents),
        ("obs_size", env.get_obs_size()),
        ("state_size", env.get_state_size()),
        ("obs_entity_shape", optional_env_value(env, "get_obs_entity_shape")),
        ("state_entity_shape", optional_env_value(env, "get_state_entity_shape")),
        ("obs_feature_names", optional_env_value(env, "get_obs_feature_names")),
        ("action_size", env.get_action_size()),
        ("constructor_kwargs", kwargs),
    ]
    writer.add_text(
        "environment/summary",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                f"|{key}|{markdown_table_value(value)}|" for key, value in env_summary
            )
        ),
    )
    writer.add_text(
        "environment/config",
        "```json\n%s\n```" % format_json(getattr(env, "config", {})),
    )

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    training_step = 0
    num_episodes = 0
    step = 0
    eval_index = 0
    best_metric_value = None
    last_eval_metrics = None
    while step < args.total_timesteps:
        num_episode = 0
        while num_episode < args.batch_size:
            episode = {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "states": [],
                "next_states": [],
                "terminated": [],
                "truncated": [],
                "avail_actions": [],
            }
            obs, _ = reset_env(env, seed=args.seed + num_episodes + num_episode)
            ep_reward, ep_length = 0, 0
            terminated, truncated = False, False
            while not terminated and not truncated:
                avail_action = env.get_avail_actions()
                state = env.get_state()
                with torch.no_grad():
                    actions, log_probs = actor.act(
                        torch.from_numpy(obs).float().to(device),
                        avail_action=torch.from_numpy(avail_action).bool().to(device),
                    )
                next_obs, reward, terminated, truncated, infos = env.step(
                    actions.cpu().numpy()
                )
                next_state = env.get_state()
                reward = shared_scalar_reward(reward)
                terminated = scalar_flag(terminated)
                truncated = scalar_flag(truncated)
                ep_reward += reward
                ep_length += 1
                step += 1
                episode["obs"].append(obs)
                episode["actions"].append(actions.cpu())
                episode["log_prob"].append(log_probs.cpu())
                episode["reward"].append(reward)
                episode["terminated"].append(terminated)
                episode["truncated"].append(truncated)
                episode["avail_actions"].append(avail_action)
                episode["states"].append(state)
                episode["next_states"].append(next_state)

                obs = next_obs

            rb.add(episode)
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == "smaclite":
                ep_stats.append(infos)
            num_episode += 1
        num_episodes += args.batch_size
        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episodes, step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []
        ## Collate episodes in buffer into single batch
        (
            b_obs,
            b_actions,
            b_log_probs,
            b_reward,
            b_states,
            b_next_states,
            b_avail_actions,
            b_terminated,
            b_truncated,
            b_mask,
        ) = rb.get_batch()

        # Compute TD(lambda) returns and advantages.
        # Only true termination clears bootstrap; time truncation uses V(next_state).
        return_lambda = torch.zeros_like(b_actions).float().to(device)
        advantages = torch.zeros_like(b_actions).float().to(device)
        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = int(b_mask[ep_idx].sum().item())
                if ep_len == 0:
                    continue

                final_t = ep_len - 1
                if bool(b_terminated[ep_idx, final_t].item()):
                    last_return_lambda = torch.zeros(env.n_agents, device=device)
                else:
                    last_return_lambda = (
                        critic(x=b_next_states[ep_idx, final_t])
                        .squeeze(-1)
                        .expand(env.n_agents)
                    )

                for t in reversed(range(ep_len)):
                    if bool(b_terminated[ep_idx, t].item()):
                        next_value = torch.zeros(env.n_agents, device=device)
                        last_return_lambda = torch.zeros(env.n_agents, device=device)
                    elif t == final_t:
                        next_value = last_return_lambda
                    else:
                        next_value = (
                            critic(x=b_states[ep_idx, t + 1])
                            .squeeze(-1)
                            .expand(env.n_agents)
                        )

                    current_value = (
                        critic(x=b_states[ep_idx, t]).squeeze(-1).expand(env.n_agents)
                    )
                    return_lambda[ep_idx, t] = last_return_lambda = b_reward[
                        ep_idx, t
                    ] + args.gamma * (
                        args.td_lambda * last_return_lambda
                        + (1 - args.td_lambda) * next_value
                    )
                    advantages[ep_idx, t] = return_lambda[ep_idx, t] - current_value

        if args.normalize_advantage:
            adv_valid = advantages[b_mask]
            adv_mu = adv_valid.mean()
            adv_std = adv_valid.std(unbiased=False)
            advantages[b_mask] = (adv_valid - adv_mu) / (adv_std + 1e-8)
        if args.normalize_return:
            ret_valid = return_lambda[b_mask]
            ret_mu = ret_valid.mean()
            ret_std = ret_valid.std(unbiased=False)
            return_lambda[b_mask] = (ret_valid - ret_mu) / (ret_std + 1e-8)
        # training loop
        actor_losses = []
        critic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []
        for _ in range(args.epochs):
            actor_loss = 0
            critic_loss = 0
            entropies = 0
            kl_divergence = 0
            clipped_ratio = 0
            for t in range(b_obs.size(1)):
                # policy gradient (PG) loss
                ## PG: compute the ratio:
                current_logits = actor.logits(
                    x=b_obs[:, t], avail_action=b_avail_actions[:, t]
                )
                current_dist = Categorical(logits=current_logits)
                current_logprob = current_dist.log_prob(b_actions[:, t])

                log_ratio = current_logprob - b_log_probs[:, t]
                ratio = torch.exp(log_ratio)
                ## Compute PG the loss
                pg_loss1 = advantages[:, t] * ratio
                pg_loss2 = advantages[:, t] * torch.clamp(
                    ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
                )
                pg_loss = (
                    torch.min(pg_loss1[b_mask[:, t]], pg_loss2[b_mask[:, t]])
                    .mean(dim=-1)
                    .sum()
                )

                # Compute entropy bonus
                entropy_loss = current_dist.entropy()[b_mask[:, t]].mean(dim=-1).sum()
                entropies += entropy_loss
                actor_loss += -pg_loss - args.entropy_coef * entropy_loss

                # Compute the value loss
                current_values = critic(x=b_states[:, t]).expand(-1, env.n_agents)
                value_loss = F.mse_loss(
                    current_values[b_mask[:, t]], return_lambda[:, t][b_mask[:, t]]
                ) * (b_mask[:, t].sum())
                critic_loss += value_loss

                # track kl distance
                b_kl_divergence = (
                    ((ratio - 1) - log_ratio)[b_mask[:, t]].mean(dim=-1).sum()
                )
                kl_divergence += b_kl_divergence
                clipped_ratio += (
                    ((ratio - 1.0).abs() > args.ppo_clip)[b_mask[:, t]]
                    .float()
                    .mean(dim=-1)
                    .sum()
                )

            actor_loss /= b_mask.sum()
            critic_loss /= b_mask.sum()
            entropies /= b_mask.sum()
            kl_divergence /= b_mask.sum()
            clipped_ratio /= b_mask.sum()

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
            critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=args.clip_gradients
                )
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), max_norm=args.clip_gradients
                )
            actor_optimizer.step()
            critic_optimizer.step()
            training_step += 1

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies_bonuses.append(entropies.item())
            kl_divergences.append(kl_divergence.item())
            actor_gradients.append(actor_gradient.item())
            critic_gradients.append(critic_gradient.item())
            clipped_ratios.append(clipped_ratio.item())

        writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients), step)
        writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
        writer.add_scalar("train/num_updates", training_step, step)

        if (training_step / args.epochs) % args.eval_steps == 0:
            actor.eval()
            eval_obs, _ = reset_env(eval_env, seed=args.seed + 10000)
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    actions, _ = actor.act(
                        torch.from_numpy(eval_obs).float().to(device),
                        avail_action=torch.from_numpy(eval_env.get_avail_actions())
                        .bool()
                        .to(device),
                        deterministic=True,
                    )
                next_obs_, reward, terminated, truncated, infos = eval_env.step(
                    actions.cpu().numpy()
                )
                terminated = scalar_flag(terminated)
                truncated = scalar_flag(truncated)
                current_reward += shared_scalar_reward(reward)
                current_ep_length += 1
                eval_obs = next_obs_
                if terminated or truncated:
                    eval_obs, _ = reset_env(
                        eval_env, seed=args.seed + 10000 + eval_ep + 1
                    )
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
            eval_metrics = {
                "eval/ep_reward": float(np.mean(eval_ep_reward)),
                "eval/std_ep_reward": float(np.std(eval_ep_reward)),
                "eval/ep_length": float(np.mean(eval_ep_length)),
            }
            if args.env_type == "smaclite":
                eval_metrics["eval/battle_won"] = float(
                    np.mean([info["battle_won"] for info in eval_ep_stats])
                )
            for metric_name, metric_value in eval_metrics.items():
                writer.add_scalar(metric_name, metric_value, step)

            eval_index += 1
            last_eval_metrics = eval_metrics
            if args.save_checkpoints:
                if args.checkpoint_best_metric not in eval_metrics:
                    available_metrics = ", ".join(sorted(eval_metrics))
                    raise ValueError(
                        "checkpoint_best_metric must match an evaluation metric, "
                        f"got {args.checkpoint_best_metric!r}. "
                        f"Available metrics: {available_metrics}"
                    )

                current_metric_value = eval_metrics[args.checkpoint_best_metric]
                is_best = (
                    best_metric_value is None
                    or (
                        args.checkpoint_best_mode == "max"
                        and current_metric_value > best_metric_value
                    )
                    or (
                        args.checkpoint_best_mode == "min"
                        and current_metric_value < best_metric_value
                    )
                )
                if is_best:
                    best_metric_value = current_metric_value
                    save_actor_checkpoint(
                        run_dir / "checkpoint_best.pt",
                        actor,
                        args,
                        kwargs,
                        step,
                        training_step,
                        num_episodes,
                        eval_index,
                        "best",
                        eval_metrics,
                        best_metric_value,
                    )
                if (
                    args.checkpoint_interval > 0
                    and eval_index % args.checkpoint_interval == 0
                ):
                    interval_path = run_dir / f"checkpoint_eval_{eval_index:06d}.pt"
                    if is_best:
                        shutil.copy(run_dir / "checkpoint_best.pt", interval_path)
                    else:
                        save_actor_checkpoint(
                            interval_path,
                            actor,
                            args,
                            kwargs,
                            step,
                            training_step,
                            num_episodes,
                            eval_index,
                            "eval",
                            eval_metrics,
                            best_metric_value,
                        )
            actor.train()

    if args.save_checkpoints:
        save_actor_checkpoint(
            run_dir / "checkpoint_final.pt",
            actor,
            args,
            kwargs,
            step,
            training_step,
            num_episodes,
            eval_index,
            "final",
            last_eval_metrics,
            best_metric_value,
        )

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
