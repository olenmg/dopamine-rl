from typing import Union

import numpy as np
import torch

from rl_algo.algorithm import ValueIterationAlgorithm
from utils.replay_buffer import ReplayBuffer
from utils.config import TrainConfig, C51Config


class C51(ValueIterationAlgorithm):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: C51Config,
        render: bool = False
    ):
        super().__init__(train_config=train_config, algo_config=algo_config, render=render)
        assert isinstance(algo_config, C51Config), "Given config instance should be a C51Config class."

        # C51 configurations
        self.gamma = algo_config.discount_rate
        self.tau = algo_config.soft_update_rate

        self.learning_starts = algo_config.learning_starts
        self.train_freq = algo_config.train_freq
        self.target_update_freq = algo_config.target_update_freq

        self.memory = ReplayBuffer(size=algo_config.buffer_size)
        self.buffer_cnt = 0

        self.value_range = torch.linspace(
            algo_config.v_min,
            algo_config.v_max,
            algo_config.n_atom,
            dtype=torch.float32
        ).to(self.device)
        self.v_min = algo_config.v_min
        self.v_max = algo_config.v_max
        self.n_atom = algo_config.n_atom
        self.v_step = (self.v_max - self.v_min) / (self.n_atom - 1)

    # Update online network with samples in the replay memory. 
    def update_network(self) -> None:
        # Do sampling from the buffer
        obses, actions, rewards, next_obses, dones = tuple(map(
            lambda x: torch.from_numpy(x).to(self.device),
            self.memory.sample(self.batch_size)
        ))
        obses, rewards, next_obses = obses.float(), rewards.float(), next_obses.float()
        # obses, next_obses         : (B, state_len, 84, 84)
        # actions, rewards, dones   : (B, )

        # Get q-value from the target network
        with torch.no_grad():
            # Calculate the estimated value distribution with target networks
            next_q_dist = self.target_net(next_obses) # (B, n_act, n_atom)
            opt_acts = torch.sum(
                next_q_dist * self.value_range.view(1, 1, -1), dim=-1
            ).argmax(dim=-1).view(-1, 1, 1) # (B, 1, 1)
            est_q_dist = next_q_dist.gather(
                1, opt_acts.expand(-1, 1, self.n_atom)
            ).squeeze() # (B, n_atom)

            # Calculate the y-distribution with estimated value distribution
            next_v_range = rewards.view(-1, 1) + \
                self.gamma * self.value_range.view(1, -1) * (~dones).view(-1, 1)
            next_v_range = torch.clamp(next_v_range, self.v_min, self.v_max) # (B, n_atom)
            next_v_pos = (next_v_range - self.v_min) / self.v_step # (B, n_atom)

            lb = torch.floor(next_v_pos).long() # (B, n_atom)
            ub = torch.ceil(next_v_pos).long() # (B, n_atom)
            
            y_dist = torch.zeros_like(est_q_dist) # (B, n_atom)
            y_dist.scatter_add_(1, lb, est_q_dist * (ub - next_v_pos)) # (B, n_atom)
            y_dist.scatter_add_(1, ub, est_q_dist * (next_v_pos - lb)) # (B, n_atom)
            
        # Calculate the predicted value distribution with pred. networks
        pred_dist = self.pred_net(obses).gather(
            1, actions.view(-1, 1, 1).expand(-1, 1, self.n_atom)
        ).squeeze() # (B, n_atom)

        # Forward pass & Backward pass
        self.pred_net.train()
        self.optimizer.zero_grad()
        loss = self.criterion(pred_dist, y_dist)
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
                action_dist = self.pred_net(obses.to(self.device)) # (n_envs, n_actions, n_atom)
                action_value = torch.sum(action_dist * self.value_range.view(1, 1, -1), dim=-1) # (n_envs, n_actions)
                action = torch.argmax(action_value, dim=-1).cpu().numpy() # (n_envs, )
        else:
            action = self.rng.choice(self.n_act, size=(self.n_envs, ))

        return action
