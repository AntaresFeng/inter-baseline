import copy
import json
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import sys

import numpy as np
import torch
import tyro
from torch.distributions.categorical import Categorical


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mappo import (  # noqa: E402
    Args as TrainArgs,
    environment,
    make_actor,
    reset_env,
    scalar_flag,
    shared_scalar_reward,
)


@dataclass
class Args:
    checkpoint_path: str
    """Path to checkpoint_best.pt or checkpoint_final.pt."""
    num_episodes: int = 3
    """Number of evaluation episodes."""
    seed: int | None = None
    """Evaluation seed. Defaults to the seed saved in the checkpoint."""
    device: str = "cpu"
    """Device used to run the actor."""
    deterministic: bool = True
    """Use greedy actions instead of sampling."""
    record_video: bool = True
    """Save one MP4 per episode."""
    save_trace: bool = True
    """Save a JSONL trace with one row per environment step."""
    output_dir: str | None = None
    """Directory for videos and traces. Defaults to the checkpoint directory."""
    trace_path: str | None = None
    """Trace JSONL path. Defaults to <output_dir>/<checkpoint_stem>_trace.jsonl."""
    fps: int = 15
    """Video fps. Use <= 0 to infer from environment metadata when available."""
    max_steps: int | None = None
    """Optional hard cap per episode."""


def ensure_imageio():
    if find_spec("imageio") is None or find_spec("imageio_ffmpeg") is None:
        raise RuntimeError(
            "Video output requires imageio and imageio-ffmpeg. "
            "Install them with: uv add imageio imageio-ffmpeg"
        )
    import imageio.v2 as imageio

    return imageio


def train_args_from_checkpoint(checkpoint_args):
    train_args = TrainArgs()
    for key, value in checkpoint_args.items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    return train_args


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return repr(value)


def write_trace(trace_file, payload):
    trace_file.write(json.dumps(payload, ensure_ascii=False, default=json_default))
    trace_file.write("\n")
    trace_file.flush()


def infer_fps(env, requested_fps):
    if requested_fps > 0:
        return int(requested_fps)
    for candidate in (env, getattr(env, "env", None)):
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("render_fps"):
            return int(metadata["render_fps"])
    config = getattr(env, "config", {})
    if isinstance(config, dict) and config.get("simulation_frequency"):
        return int(config["simulation_frequency"])
    return 15


def normalize_frame(frame):
    if frame is None:
        raise RuntimeError("env.render() returned None. Is render_mode='rgb_array'?")
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8, copy=False)


def render_frame(env):
    return normalize_frame(env.render())


def main():
    args = tyro.cli(Args)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    checkpoint_args = checkpoint.get("args", {})
    seed = args.seed if args.seed is not None else int(checkpoint_args.get("seed", 1))

    env_config = copy.deepcopy(checkpoint.get("env_config", {}))
    if args.record_video:
        env_config["render_mode"] = "rgb_array"

    env = environment(
        env_type=checkpoint_args.get("env_type", "highway"),
        env_name=checkpoint_args.get("env_name", "intersection-multi-agent-v1"),
        env_family=checkpoint_args.get("env_family", "mpe"),
        agent_ids=checkpoint_args.get("agent_ids", True),
        kwargs=env_config,
    )

    device = torch.device(args.device)
    actor = make_actor(train_args_from_checkpoint(checkpoint_args), env).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    imageio = ensure_imageio() if args.record_video else None
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    if args.record_video or args.save_trace:
        output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = (
        Path(args.trace_path)
        if args.trace_path
        else output_dir / f"{checkpoint_path.stem}_trace.jsonl"
    )
    fps = infer_fps(env, args.fps)

    rewards = []
    lengths = []
    trace_file = None
    try:
        if args.save_trace:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = trace_path.open("w", encoding="utf-8")
            print(f"trace={trace_path}")

        for episode in range(args.num_episodes):
            obs, _ = reset_env(env, seed=seed + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0
            frame_count = 0

            writer = None
            video_path = None
            if imageio is not None:
                video_path = output_dir / f"{checkpoint_path.stem}_ep{episode:03d}.mp4"
                writer = imageio.get_writer(
                    str(video_path), fps=fps, macro_block_size=1
                )
                writer.append_data(render_frame(env))
                frame_count += 1

            try:
                while not terminated and not truncated:
                    if args.max_steps is not None and episode_length >= args.max_steps:
                        truncated = True
                        break

                    avail_actions = env.get_avail_actions()
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs).float().to(device)
                        avail_tensor = torch.from_numpy(avail_actions).bool().to(device)
                        logits = actor.logits(obs_tensor, avail_tensor)
                        distribution = Categorical(logits=logits)
                        if args.deterministic:
                            actions = torch.argmax(logits, dim=-1)
                        else:
                            actions = distribution.sample()
                        log_probs = distribution.log_prob(actions)
                        probs = distribution.probs
                        entropies = distribution.entropy()

                    next_obs, reward, terminated, truncated, info = env.step(
                        actions.detach().cpu().numpy()
                    )
                    reward = shared_scalar_reward(reward)
                    terminated = scalar_flag(terminated)
                    truncated = scalar_flag(truncated)
                    episode_reward += reward
                    episode_length += 1

                    if trace_file is not None:
                        write_trace(
                            trace_file,
                            {
                                "episode": episode,
                                "episode_seed": seed + episode,
                                "step": episode_length,
                                "reward": reward,
                                "episode_reward": episode_reward,
                                "terminated": terminated,
                                "truncated": truncated,
                                "actions": actions,
                                "log_probs": log_probs,
                                "entropies": entropies,
                                "probs": probs,
                                "logits": logits,
                                "avail_actions": avail_actions,
                                "info": info,
                            },
                        )

                    if writer is not None:
                        writer.append_data(render_frame(env))
                        frame_count += 1
                    obs = next_obs
            finally:
                if writer is not None:
                    writer.close()

            rewards.append(episode_reward)
            lengths.append(episode_length)
            done_reason = "terminated" if terminated else "truncated"
            video_msg = "" if video_path is None else f" video={video_path}"
            print(
                f"episode={episode} reward={episode_reward:.3f} "
                f"length={episode_length} frames={frame_count} "
                f"done={done_reason}{video_msg}"
            )

        print(
            "summary "
            f"reward_mean={np.mean(rewards):.3f} "
            f"reward_std={np.std(rewards):.3f} "
            f"length_mean={np.mean(lengths):.3f}"
        )
    finally:
        if trace_file is not None:
            trace_file.close()
        env.close()


if __name__ == "__main__":
    main()
