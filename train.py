from typing import List

import torch
import torch.nn as nn
import gymnasium as gym

from algorithms import DQN, C51
from utils.config import TrainConfig, DQNConfig, C51Config
from utils.policy_networks import MLPNet, ConvNet
from utils.plot import plot_train_result


def train_dqn(
    train_config: TrainConfig
) -> List[int]:
    sample_env = gym.make(train_config.env_id)
    policy_network = MLPNet(
        input_size=sample_env.observation_space.shape[0],
        output_size=sample_env.action_space.n,
        hidden_sizes=[128, 128]
    )
    sample_env.close()
    del sample_env

    dqn_config = DQNConfig(
        policy_network=policy_network,
        eps_start=0.5,
        eps_end=0.01,
        eps_decay=0.9995,
        discount_rate=0.98,
        buffer_size=100000,
        learning_starts=512,
        train_freq=1,
        target_update_freq=512
    )

    dqn = DQN(
        train_config=train_config,
        algo_config=dqn_config
    )
    result = dqn.train()
    
    return result


if __name__ == "__main__":
    train_config = TrainConfig(
        run_name="test",
        env_id="CartPole-v1",
        n_envs=4,
        state_len=1, 
        random_seed=42,
        optim_cls=torch.optim.Adam,
        optim_kwargs={'lr': 0.0003},
        loss_fn=nn.SmoothL1Loss(),
        batch_size=128,
        train_step=int(1e5),
        save_freq=-1,
        device="cpu",
        verbose=True
    )

    result_infos = train_dqn(train_config=train_config)
    plot_train_result(
        result=[info["r"] for info in result_infos],
        label="DQN",
        alpha=0.9
    )
