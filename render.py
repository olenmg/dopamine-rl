import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from rl_algo import DQN, C51, QRDQN
from utils.config import TrainConfig, DQNConfig, C51Config, QRConfig

ALGO_CONFIG = {
    'DQN': DQNConfig, 'C51': C51Config, 'QRDQN': QRConfig
}
ALGO = {
    'DQN': DQN, 'C51': C51, "QRDQN": QRDQN
}

def get_render_frames(model, env, n_step=10000):
    total_reward = 0
    done_counter = 0
    frames = []

    obs, _ = env.reset()
    for _ in range(n_step):
        # Render into buffer. 
        frames.append(env.render())
        action = model.predict(obs, eps=0.01)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            done_counter += 1
            obs, _ = env.reset()
        else:
            obs = next_obs

        if done_counter == 2:
            break
    env.close()
    print(f"Total Reward: {total_reward:.2f}")
    return frames

def display_frames_as_gif(frames, fname="result.gif"):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
        
    ani = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    ani.save(fname, writer='pillow', fps=30)

def render(args):
    if args.log_path:
        dir_path = os.path.join("./results", args.log_path)
        with open(os.path.join(dir_path, "train_cfg.json"), "r") as f:
            train_config = TrainConfig(**json.load(f))
        with open(os.path.join(dir_path, "algo_cfg.json"), "r") as f:
            algo_config = ALGO_CONFIG[train_config.algo](**json.load(f))
    else:
        with open(os.path.join("configs/train_configs", args.train_cfg), "r") as f:
            train_config = TrainConfig(**json.load(f))
        with open(os.path.join("configs/algo_configs", args.algo_cfg), "r") as f:
            algo_config = ALGO_CONFIG[train_config.algo](**json.load(f))
        dir_path = os.path.join("./results", train_config.run_name)

    train_config.n_envs = 1
    f_prefix = "best_" if args.use_best else ""

    model = ALGO[train_config.algo](
        train_config=train_config,
        algo_config=algo_config,
        render=True
    )
    model.load_model(
        pred_net_fname=os.path.join(dir_path, f_prefix + "pred_net.pt"),
        target_net_fname=os.path.join(dir_path, f_prefix + "target_net.pt"),
    )

    env = model.env
    if hasattr(env, "render_mode"):
        env.unwrapped.render_mode = "rgb_array"

    frames = get_render_frames(
        model=model,
        env=env,
        n_step=10000
    )

    display_frames_as_gif(
        frames=frames,
        fname=os.path.join(dir_path, "video.gif") if args.log_path \
            else os.path.join("results", train_config.run_name, "video.gif")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for rendering")
    parser.add_argument(
        '--log-path', type=str, default=""
    )
    parser.add_argument(
        '--train-cfg', type=str, default="",
        help="Path of train config"
    )
    parser.add_argument(
        '--algo-cfg', type=str, default="",
        help="Path of algorithm config"
    )
    parser.add_argument(
        '--use-best', action="store_true",
        help="Whether to use the best model"
    )
    args = parser.parse_args()
    render(args)
