import os
import json
import argparse

from rl_algo import DQN, C51, QRDQN
from utils.config import TrainConfig, DQNConfig, C51Config, QRConfig
from utils.plot import plot_train_result
from render import render

ALGO_CONFIG = {
    'DQN': DQNConfig, 'C51': C51Config, 'QRDQN': QRConfig
}
ALGO = {
    'DQN': DQN, 'C51': C51, "QRDQN": QRDQN
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument(
        '--algo', type=str,
        help="Algorithm", choices=['DQN', 'C51', 'QRDQN']
    )
    parser.add_argument(
        '--train-cfg', type=str,
        help="Path of train config"
    )
    parser.add_argument(
        '--algo-cfg', type=str,
        help="Path of algorithm config"
    )
    args = parser.parse_args()

    with open(os.path.join("configs/train_configs", args.train_cfg), "r") as f:
        train_config = TrainConfig(**json.load(f))
    with open(os.path.join("configs/algo_configs", args.algo_cfg), "r") as f:
        algo_config = ALGO_CONFIG[args.algo](**json.load(f))

    model = ALGO[args.algo](
        train_config=train_config,
        algo_config=algo_config
    )
    result = model.train()
    plot_train_result(
        result=[info["r"] for info in result],
        label=train_config.run_name,
        save_path=os.path.join("results", train_config.run_name, f"{train_config.run_name}.png"),
        alpha=0.9
    )
    args.log_path = ""
    render(args)
