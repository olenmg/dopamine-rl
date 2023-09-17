from typing import Union

import numpy as np
import torch
from gymnasium.wrappers.frame_stack import LazyFrames

from rl_algo.algorithm import ValueIterationAlgorithm
from utils.replay_buffer import ReplayBuffer
from utils.config import TrainConfig, MGDQNConfig


class MGDQN(ValueIterationAlgorithm):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: MGDQNConfig,
        render: bool = False
    ):
        super().__init__(train_config=train_config, algo_config=algo_config, render=render)
        assert isinstance(algo_config, MGDQNConfig), "Given config instance should be a MGDQNConfig class."

        # DQN configurations
        self.gamma_min = algo_config.gamma_min
        self.gamma_max = algo_config.gamma_max
        self.gamma_n = algo_config.gamma_n
        self.gamma_range = torch.linspace(
            algo_config.gamma_min,
            algo_config.gamma_max,
            algo_config.gamma_n,
            dtype=torch.float32
        ).to(self.device)
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
        actions, rewards = actions.long(), rewards.float()
        if self.reward_clipping:
            rewards = torch.clamp(rewards, -1, 1)

        # Get q-value from the target network
        with torch.no_grad():
            next_q_vals = self.target_net(next_obses) # (B, n_act, n_gamma)
            next_q_vals = self.gamma_range.view(1, 1, -1) * next_q_vals # (B, n_act, n_gamma)
            opt_acts = next_q_vals.mean(dim=-1).argmax(dim=-1).view(-1, 1, 1) # (B, 1, 1)
            next_q_val = next_q_vals.gather(1, opt_acts.expand(-1, 1, self.gamma_n)).squeeze(1) # (B, n_gamma)
            y = rewards.unsqueeze(-1) + (~dones).unsqueeze(-1) * next_q_val # ~dones == (1 - dones), (B, n_gamma)

        # Get predicted Q-value
        pred = self.pred_net(obses).gather(
            1, actions.view(-1, 1, 1).expand(-1, 1, self.gamma_n)
        ).squeeze() # (N, n_gamma)

        # Forward pass & Backward pass
        self.optimizer.zero_grad()
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
        if isinstance(obses, LazyFrames):
            obses = obses[:]
        if isinstance(obses, list):
            obses = np.array(list)
        if isinstance(obses, np.ndarray):
            obses = torch.from_numpy(obses)

        # Epsilon-greedy
        if self.rng.random() >= eps:
            self.pred_net.eval()
            with torch.no_grad():
                action_value = torch.mean(
                    self.gamma_range.view(1, 1, -1) * self.pred_net(obses.to(self.device)),
                    dim=-1
                ) # (n_envs, n_actions)
                action = torch.argmax(action_value, dim=-1).cpu().numpy() # (n_envs, )
        else:
            action = self.rng.choice(self.n_act, size=(self.n_envs, ))

        return action.squeeze()
