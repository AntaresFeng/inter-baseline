import json
import copy
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont
from torch.distributions.categorical import Categorical

from mappo import (
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
    """Path to a MAPPO actor checkpoint."""
    num_episodes: int = 3
    """Number of evaluation episodes."""
    seed: int | None = None
    """Evaluation seed; defaults to the checkpoint seed."""
    device: str = "cpu"
    """Device (cpu, cuda, mps)."""
    deterministic: bool = True
    """Use greedy actions if True; sample from the policy if False."""
    display: bool = False
    """Display the annotated RGB frames in a pygame window."""
    record_video: bool = True
    """Record annotated MP4 videos next to the checkpoint."""
    fps: int = 15
    """Video and display frame rate; use 0 to follow env.metadata render_fps."""
    overlay: bool = True
    """Draw environment and policy diagnostics over the rendered frame."""


def load_font(size: int):
    for font_path in [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def to_float(value):
    return float(np.asarray(value, dtype=np.float32))


def tensor_list(tensor):
    return tensor.detach().cpu().numpy().tolist()


def action_names(env):
    names = []
    action_types = getattr(getattr(env, "action_type", None), "agents_action_types", [])
    for action_type in action_types:
        indexes = getattr(action_type, "actions_indexes", {})
        reverse = {int(value): str(key) for key, value in indexes.items()}
        names.append(reverse)
    return names


def action_label(action_names_by_agent, agent_idx, action):
    if agent_idx < len(action_names_by_agent):
        name = action_names_by_agent[agent_idx].get(int(action))
        if name is not None:
            return f"{int(action)}:{name}"
    return str(int(action))


def vehicle_rows(env, info):
    vehicles = getattr(env, "controlled_vehicles", [])
    agents_rewards = info.get("agents_rewards", ())
    agents_arrived = info.get("agents_arrived", ())
    agents_active = info.get("agents_active", ())
    rows = []
    for idx, vehicle in enumerate(vehicles):
        reward = agents_rewards[idx] if idx < len(agents_rewards) else None
        arrived = agents_arrived[idx] if idx < len(agents_arrived) else None
        active = agents_active[idx] if idx < len(agents_active) else None
        rows.append(
            {
                "agent": idx,
                "speed": to_float(getattr(vehicle, "speed", 0.0)),
                "crashed": bool(getattr(vehicle, "crashed", False)),
                "arrived": None if arrived is None else bool(arrived),
                "active": None if active is None else bool(active),
                "reward": None if reward is None else to_float(reward),
                "position": np.asarray(getattr(vehicle, "position", [])).tolist(),
                "heading": to_float(getattr(vehicle, "heading", 0.0)),
            }
        )
    return rows


def truncate(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compact_bool(value):
    if value is None:
        return "N/A"
    return "T" if bool(value) else "F"


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_panel(image, lines, xy, max_width, font, title=None):
    if title is not None:
        lines = [title] + lines
    lines = [truncate(line, max(20, max_width // 7)) for line in lines]
    draw = ImageDraw.Draw(image, "RGBA")
    line_height = text_size(draw, "Ag", font)[1] + 5
    padding = 8
    widths = [text_size(draw, line, font)[0] for line in lines] or [0]
    panel_width = min(max_width, max(widths) + padding * 2)
    panel_height = len(lines) * line_height + padding * 2
    x, y = xy
    draw.rounded_rectangle(
        (x, y, x + panel_width, y + panel_height),
        radius=6,
        fill=(0, 0, 0, 165),
        outline=(255, 255, 255, 70),
    )
    for idx, line in enumerate(lines):
        fill = (
            (255, 231, 150, 255)
            if title is not None and idx == 0
            else (245, 245, 245, 255)
        )
        draw.text(
            (x + padding, y + padding + idx * line_height), line, font=font, fill=fill
        )
    return panel_width, panel_height


def format_prob_line(agent_idx, probs, action_names_by_agent, max_actions=6):
    if len(probs) <= max_actions:
        order = np.arange(len(probs))
    else:
        order = np.argsort(np.asarray(probs))[::-1][:max_actions]
    parts = []
    for action_idx in order:
        label = action_label(action_names_by_agent, agent_idx, action_idx)
        parts.append(f"{label}={probs[action_idx]:.2f}")
    return "p " + " ".join(parts)


def annotate_frame(frame, overlay_data):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = (
        Image.fromarray(frame.astype(np.uint8))
        .convert("RGB")
        .resize((frame.shape[1], frame.shape[0]))
    )
    font = load_font(13)
    width, height = image.size

    status_lines = [
        f"ckpt: {overlay_data['checkpoint_name']}",
        f"episode={overlay_data['episode']} step={overlay_data['step']}",
        f"ep_reward={overlay_data['episode_reward']:.3f} reward={overlay_data['reward']:.3f}",
        f"terminated={overlay_data['terminated']} truncated={overlay_data['truncated']}",
    ]
    draw_panel(image, status_lines, (8, 8), max(300, width // 2), font, "Evaluation")

    env_lines = []
    for row in overlay_data["vehicles"]:
        env_lines.append(
            "a{agent} v={speed:.2f} r={reward} "
            "arr={arrived} act={active} cr={crashed}".format(
                agent=row["agent"],
                speed=row["speed"],
                reward="N/A" if row["reward"] is None else f"{row['reward']:.3f}",
                arrived=compact_bool(row["arrived"]),
                active=compact_bool(row["active"]),
                crashed=compact_bool(row["crashed"]),
            )
        )
    _, env_panel_height = draw_panel(
        image,
        env_lines or ["no controlled vehicles"],
        (8, max(8, height - (len(env_lines) + 2) * 24)),
        max(340, width // 2),
        font,
        "Environment",
    )

    policy_lines = []
    for idx, action in enumerate(overlay_data["actions"]):
        action_name = action_label(overlay_data["action_names"], idx, action)
        policy_lines.append(
            f"a{idx} act={action_name} logp={overlay_data['log_probs'][idx]:.3f} "
            f"H={overlay_data['entropies'][idx]:.3f} V=N/A"
        )
        policy_lines.append(
            "  "
            + format_prob_line(
                idx,
                overlay_data["probs"][idx],
                overlay_data["action_names"],
            )
        )
    policy_width = max(360, min(620, int(width * 0.58)))
    policy_x = max(8, width - policy_width - 8)
    draw_panel(image, policy_lines, (policy_x, 8), policy_width, font, "MAPPO Policy")

    if env_panel_height > height:
        return np.asarray(image, dtype=np.uint8)
    return np.asarray(image, dtype=np.uint8)


def ensure_imageio():
    if find_spec("imageio") is None or find_spec("imageio_ffmpeg") is None:
        raise RuntimeError(
            "Video output requires imageio and imageio-ffmpeg. "
            "Install them with: uv pip install imageio imageio-ffmpeg"
        )
    import imageio.v2 as imageio

    return imageio


def create_display(frame_shape, fps):
    if find_spec("pygame") is None:
        raise RuntimeError("Display requires pygame.")
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((frame_shape[1], frame_shape[0]))
    pygame.display.set_caption("MAPPO checkpoint evaluation")
    clock = pygame.time.Clock()
    return pygame, screen, clock, fps


def show_frame(display_state, frame):
    pygame, screen, clock, fps = display_state
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(fps)
    return True


class EvaluationFrameRecorder:
    def __init__(self, env, writer, display_state, overlay, get_overlay_data):
        self.env = env
        self.writer = writer
        self.display_state = display_state
        self.overlay = overlay
        self.get_overlay_data = get_overlay_data
        self.frames_per_sec = None
        self.closed_by_user = False
        self.frame_count = 0

    def _capture_frame(self):
        if self.env.viewer is None:
            return
        self.env.viewer.display()
        if not self.env.viewer.offscreen:
            self.env.viewer.handle_events()
        frame = self.env.viewer.get_image()
        if frame is None:
            return
        overlay_data = self.get_overlay_data()
        output_frame = (
            annotate_frame(frame, overlay_data)
            if self.overlay
            else np.asarray(frame, dtype=np.uint8)
        )
        if self.writer is not None:
            self.writer.append_data(output_frame)
        if self.display_state is not None and not self.closed_by_user:
            self.closed_by_user = not show_frame(self.display_state, output_frame)
        self.frame_count += 1


def write_trace(trace_file, payload):
    trace_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    trace_file.flush()


def build_trace_payload(overlay_data):
    return {
        "episode": overlay_data["episode"],
        "step": overlay_data["step"],
        "episode_reward": overlay_data["episode_reward"],
        "reward": overlay_data["reward"],
        "terminated": overlay_data["terminated"],
        "truncated": overlay_data["truncated"],
        "actions": overlay_data["actions"],
        "action_names": [
            action_label(overlay_data["action_names"], i, action)
            for i, action in enumerate(overlay_data["actions"])
        ],
        "log_probs": overlay_data["log_probs"],
        "entropies": overlay_data["entropies"],
        "probs": overlay_data["probs"],
        "logits": overlay_data["logits"],
        "values": None,
        "vehicles": overlay_data["vehicles"],
    }


def training_args_from_checkpoint(checkpoint_args):
    train_args = TrainArgs()
    for key, value in checkpoint_args.items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    return train_args


def main():
    args = tyro.cli(Args)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    checkpoint_args = checkpoint.get("args", {})
    seed = args.seed if args.seed is not None else int(checkpoint_args.get("seed", 1))
    device = torch.device(args.device)
    env_config = copy.deepcopy(checkpoint.get("env_config", {}))
    env_config["render_mode"] = "rgb_array"

    env = environment(
        env_type=checkpoint_args.get("env_type", "highway"),
        env_name=checkpoint_args.get("env_name", "intersection-multi-agent-v1"),
        env_family=checkpoint_args.get("env_family", "mpe"),
        agent_ids=checkpoint_args.get("agent_ids", True),
        kwargs=env_config,
    )
    actor = make_actor(training_args_from_checkpoint(checkpoint_args), env).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    output_root = checkpoint_path.parent
    imageio = ensure_imageio() if args.record_video else None

    trace_file = None
    trace_path = output_root / f"{checkpoint_path.stem}_trace.jsonl"
    trace_file = trace_path.open("w", encoding="utf-8")
    print(f"trace={trace_path}")

    display_state = None
    ep_rewards = []
    ep_lengths = []
    fps = int(args.fps) if args.fps > 0 else int(env.config["simulation_frequency"])
    try:
        for episode in range(args.num_episodes):
            obs, _ = reset_env(env, seed=seed + episode)
            initial_frame = env.render()
            if initial_frame is None:
                raise RuntimeError("env.render() returned None for rgb_array mode.")
            if args.display and display_state is None:
                display_state = create_display(np.asarray(initial_frame).shape, fps)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_step = 0
            writer = None
            video_path = None
            if imageio is not None:
                video_path = output_root / f"{checkpoint_path.stem}_ep{episode:03d}.mp4"
                writer = imageio.get_writer(
                    str(video_path), fps=fps, macro_block_size=1
                )

            try:
                last_overlay_data = None

                def current_overlay_data():
                    if last_overlay_data is None:
                        raise RuntimeError(
                            "No overlay data available for frame capture."
                        )
                    return last_overlay_data

                recorder = EvaluationFrameRecorder(
                    env=env,
                    writer=writer,
                    display_state=display_state,
                    overlay=args.overlay,
                    get_overlay_data=current_overlay_data,
                )
                env.set_record_video_wrapper(recorder)

                while not terminated and not truncated:
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

                    last_overlay_data = {
                        "checkpoint_name": checkpoint_path.name,
                        "episode": episode,
                        "step": episode_step + 1,
                        "episode_reward": episode_reward,
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                        "actions": [int(v) for v in tensor_list(actions)],
                        "log_probs": [float(v) for v in tensor_list(log_probs)],
                        "entropies": [float(v) for v in tensor_list(entropies)],
                        "probs": tensor_list(probs),
                        "logits": tensor_list(logits),
                        "action_names": action_names(env),
                        "vehicles": vehicle_rows(env, {}),
                    }

                    next_obs, reward, terminated, truncated, info = env.step(
                        actions.detach().cpu().numpy()
                    )
                    reward = shared_scalar_reward(reward)
                    terminated = scalar_flag(terminated)
                    truncated = scalar_flag(truncated)
                    episode_reward += reward
                    episode_step += 1

                    overlay_data = {
                        "checkpoint_name": checkpoint_path.name,
                        "episode": episode,
                        "step": episode_step,
                        "episode_reward": episode_reward,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "actions": [int(v) for v in tensor_list(actions)],
                        "log_probs": [float(v) for v in tensor_list(log_probs)],
                        "entropies": [float(v) for v in tensor_list(entropies)],
                        "probs": tensor_list(probs),
                        "logits": tensor_list(logits),
                        "action_names": action_names(env),
                        "vehicles": vehicle_rows(env, info),
                    }
                    last_overlay_data = overlay_data
                    final_frame = env.render()
                    if final_frame is None:
                        raise RuntimeError(
                            "env.render() returned None for rgb_array mode."
                        )
                    recorder._capture_frame()

                    trace_payload = build_trace_payload(overlay_data)
                    trace_payload["recorded_frames"] = recorder.frame_count
                    write_trace(trace_file, trace_payload)
                    if recorder.closed_by_user:
                        args.display = False

                    obs = next_obs
            finally:
                env._record_video_wrapper = None
                env.update_metadata()
                if writer is not None:
                    writer.close()
            ep_rewards.append(episode_reward)
            ep_lengths.append(episode_step)
            done_reason = "terminated" if terminated else "truncated"
            video_msg = "" if video_path is None else f" video={video_path}"
            print(
                f"episode={episode} reward={episode_reward:.3f} "
                f"length={episode_step} frames={recorder.frame_count} "
                f"done={done_reason}{video_msg}"
            )

        print(
            "summary "
            f"reward_mean={np.mean(ep_rewards):.3f} "
            f"reward_std={np.std(ep_rewards):.3f} "
            f"length_mean={np.mean(ep_lengths):.3f}"
        )
    finally:
        if trace_file is not None:
            trace_file.close()
        if display_state is not None:
            display_state[0].quit()
        env.close()


if __name__ == "__main__":
    main()
