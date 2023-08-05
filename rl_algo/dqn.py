from typing import Union

import numpy as np
import torch

from rl_algo.algorithm import ValueIterationAlgorithm
from utils.replay_buffer import ReplayBuffer
from utils.config import TrainConfig, DQNConfig


class DQN(ValueIterationAlgorithm):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: DQNConfig,
        render: bool = False
    ):
        super().__init__(train_config=train_config, algo_config=algo_config, render=render)
        assert isinstance(algo_config, DQNConfig), "Given config instance should be a DQNConfig class."

        # DQN configurations
        self.gamma = algo_config.discount_rate
        self.tau = algo_config.soft_update_rate

        self.learning_starts = algo_config.learning_starts
        self.train_freq = algo_config.train_freq
        self.target_update_freq = algo_config.target_update_freq

        self.memory = ReplayBuffer(size=algo_config.buffer_size)
        self.buffer_cnt = 0

    # Update online network with samples in the replay memory. 
    def update_network(self) -> None:
        self.pred_net.train()

        # Do sampling from the buffer
        obses, actions, rewards, next_obses, dones = tuple(map(
            lambda x: torch.from_numpy(x).to(self.device),
            self.memory.sample(self.batch_size)
        ))
        obses, rewards, next_obses = obses.float(), rewards.float(), next_obses.float()
        actions, rewards = actions.long(), torch.sign(rewards).long()

        # Get q-value from the target network
        with torch.no_grad():
            q_val = torch.max(self.target_net(next_obses), dim=-1).values
        y = rewards + self.gamma * ~dones * q_val # ~dones == (1 - dones)

        # Forward pass & Backward pass
        self.optimizer.zero_grad()
        pred = self.pred_net(obses).gather(1, actions.unsqueeze(1)).squeeze(-1)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(
        self,
        obses: Union[list, np.ndarray],
        eps: float = -1.0
    ) -> np.ndarray:
        """
            obses: 
                Training stage : (n_envs, state_len, *, ) or (state_len, *, ) or (*, _)
                Inference stage: (batch_size, state_len, *, ) or (state_len, *, _), or (*, _)
            eps:
                -1.0 at inference stage
        """
        if isinstance(obses, list):
            obses = np.array(list)
        if isinstance(obses, np.ndarray):
            obses = torch.from_numpy(obses)
        # obses = obses.squeeze() # Squeezed when n_envs == 1 or state_len == 1

        # Epsilon-greedy
        if self.rng.random() >= eps:
            self.pred_net.eval()
            with torch.no_grad():
                action = torch.argmax(
                    self.pred_net(obses.to(self.device)), dim=-1
                ).cpu().numpy()
        else:
            action = self.rng.choice(self.n_act, size=(self.n_envs, ))

        return action
