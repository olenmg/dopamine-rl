import os
import json
import argparse

from rl_algo import *
from utils.config import *
from utils.plot import plot_train_result
from render import render


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training")
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
        algo_config = ALGO_CONFIG[train_config.algo](**json.load(f))

    model = ALGO[train_config.algo](
        train_config=train_config,
        algo_config=algo_config
    )
    result = model.train()
    plot_train_result(
        result=result,
        label=train_config.run_name,
        save_path=os.path.join("results", train_config.run_name, f"{train_config.run_name}.png"),
        alpha=0.9
    )
    args.log_path = ""
    args.use_best = True
    render(args)
