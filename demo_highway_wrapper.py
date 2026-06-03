from env.highway_wrapper import HighwayWrapper

config = {
    "arrived_reward": 10,
}


def main() -> None:
    env = HighwayWrapper(render_mode="human", **config)
    # print(env.config)
    try:
        obs, info = env.reset(seed=1)

        print(f"env={type(env).__name__}")
        print(f"n_agents={env.n_agents}")
        print(f"obs_shape={obs.shape} obs_dtype={obs.dtype}")
        print(f"state_shape={env.get_state().shape}")
        print(f"action_size={env.get_action_size()}")
        print(f"reset_info_keys={list(info)[:5]}")

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # action = env.action_space.sample()
            action = (2, 2)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"step={step} action={action} reward={reward:.3f} "
                f"terminated={terminated} truncated={truncated} "
                f"obs_shape={obs.shape}"
            )
            step += 1

        print(f"done_reason={'terminated' if terminated else 'truncated'}")
        print(f"steps={step} total_reward={total_reward:.3f}")
        print(f"step_info_keys={list(info)[:5]}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
