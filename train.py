from typing import List

import torch
import torch.nn as nn
import gymnasium as gym

from rl_algo import DQN, C51
from utils.config import TrainConfig, DQNConfig, C51Config
from utils.policy_networks import MLPNet, ConvNet
from utils.loss import SoftCrossEntropyLoss
from utils.plot import plot_train_result


def train_c51(
    train_config: TrainConfig
) -> List[int]:
    sample_env = gym.make(train_config.env_id)
    policy_network = ConvNet(
        n_actions=sample_env.action_space.n,
        state_len=train_config.state_len,
        n_atom=51
    )
    sample_env.close()
    del sample_env

    c51_config = C51Config(
        policy_network=policy_network,
        v_min=-5,
        v_max=10,
        n_atom=51,
        eps_start=0.9,
        eps_end=0.01,
        eps_decay=0.9999,
        discount_rate=0.99,
        soft_update_rate=0.01,
        buffer_size=100000,
        learning_starts=512,
        train_freq=1,
        target_update_freq=1
    )

    c51 = C51(
        train_config=train_config,
        algo_config=c51_config
    )
    result = c51.train()
    
    return result


if __name__ == "__main__":
    train_config = TrainConfig(
        run_name="test",
        env_id="ALE/Breakout-v5",
        n_envs=4,
        state_len=4,
        frame_skip=1,
        random_seed=42,
        optim_cls=torch.optim.Adam,
        optim_kwargs={'lr': 1.1e-4},
        loss_fn=SoftCrossEntropyLoss(),
        batch_size=32,
        train_step=int(1e5),
        save_freq=-1,
        device="cpu",
        verbose=True
    )

    result_infos = train_c51(train_config=train_config)
    plot_train_result(
        result=[info["r"] for info in result_infos],
        label="C51",
        alpha=0.9
    )
