from typing import Union

import numpy as np
import torch
from gymnasium.wrappers.frame_stack import LazyFrames

from rl_algo.algorithm import ValueIterationAlgorithm
from utils.replay_buffer import ReplayBuffer
from utils.config import TrainConfig, MGC51Config


class MGC51(ValueIterationAlgorithm):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: MGC51Config,
        render: bool = False
    ):
        super().__init__(train_config=train_config, algo_config=algo_config, render=render)
        assert isinstance(algo_config, MGC51Config), "Given config instance should be a MGC51Config class."

        # C51 configurations
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

        # obses, next_obses         : (B, state_len, 84, 84)
        # actions, rewards, dones   : (B, )

        # Get q-value from the target network
        with torch.no_grad():
            # Calculate the estimated value distribution with target networks
            next_q_dist = self.target_net(next_obses) # (B, n_act, gamma_n, n_atom)
            batched_votes = torch.sum(
                next_q_dist * self.value_range.view(1, 1, 1, -1), dim=-1
            ).argmax(dim=1) # (B, gamma_n)
            opt_acts = []
            for votes in batched_votes:
                opt_acts.append(torch.bincount(votes).argmax().item())
            opt_acts = torch.tensor(opt_acts, device=self.device).view(-1, 1, 1, 1) # (B, 1, 1, 1)
            est_q_dist = next_q_dist.gather(
                1, opt_acts.expand(-1, 1, self.gamma_n, self.n_atom)
            ).squeeze() # (B, gamma_n, n_atom)

            # Calculate the y-distribution with estimated value distribution
            next_v_range = rewards.view(-1, 1, 1) + \
                self.gamma_range.view(1, -1, 1) * self.value_range.view(1, 1, -1) * (~dones).view(-1, 1, 1) # (B, gamma_n, n_atom)
            next_v_range = torch.clamp(next_v_range, self.v_min, self.v_max) # (B, gamma_n, n_atom)
            next_v_pos = (next_v_range - self.v_min) / self.v_step # (B, gamma_n, n_atom)

            lb = torch.floor(next_v_pos).long() # (B, gamma_n, n_atom)
            ub = torch.ceil(next_v_pos).long() # (B, gamma_n, n_atom)
            
            y_dist = torch.zeros_like(est_q_dist) # (B, gamma_n, n_atom)
            y_dist.scatter_add_(2, lb, est_q_dist * (ub - next_v_pos)) # (B, gamma_n, n_atom)
            y_dist.scatter_add_(2, ub, est_q_dist * (next_v_pos - lb)) # (B, gamma_n, n_atom)
            
        # Calculate the predicted value distribution with pred. networks
        pred_dist = self.pred_net(obses).gather(
            1, actions.view(-1, 1, 1, 1).expand(-1, 1, self.gamma_n, self.n_atom)
        ).squeeze() # (B, gamma_n, n_atom)

        # Forward pass & Backward pass
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
                q_dist = self.pred_net(obses.to(self.device)) # ((n_envs,) n_act, gamma_n, n_atom)
                q_vals = torch.sum(
                    q_dist * self.value_range.view((1, ) * (q_dist.dim() - 1) + (-1, )),
                    dim=-1
                ) # ((n_envs,) n_act, gamma_n)
                batched_votes = q_vals.argmax(dim=-2) # ((n_envs,) gamma_n)
                if self.n_envs == 1:
                    action = torch.bincount(batched_votes).argmax().cpu().item()
                else:
                    action = []
                    for votes in batched_votes:
                        action.append(torch.bincount(votes).argmax().cpu().item())
                action = np.asarray(action) # ((n_envs,) )
        else:
            action = self.rng.choice(self.n_act, size=(self.n_envs, ))

        return action.squeeze()
