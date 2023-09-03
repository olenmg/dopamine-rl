from typing import Union, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MlpPolicy(PolicyNetwork):
    def __init__(
        self,
        algo: str,
        n_actions: int,
        input_size: int,
        hidden_sizes: List[int] = [64, ],
        state_len: int = 1,
        n_out: int = -1
    ):
        """
            n_out: If given, network will be built for C51 algorithm
        """

        super().__init__()
        self.algo = algo
        self.in_layer = nn.Linear(input_size * state_len, hidden_sizes[0])
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_shape, out_shape) for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        if n_out == -1:
            self.fc_q = nn.Linear(hidden_sizes[-1], n_actions)
        else:
            self.fc_q = nn.Linear(hidden_sizes[-1], n_actions * n_out)

        self.n_actions = n_actions
        self.n_out = n_out
        self.state_len = state_len

        self._init_weight()
    
    def forward(self, x):
        if self.state_len != 1:
            x = x.flatten(-2)
        x = F.relu(self.in_layer(x))
        for layer in self.linears:
            x = F.relu(layer(x))

        if self.algo == "DQN":
            action_value = self.fc_q(x)
        elif self.algo == "C51":
            action_value = F.softmax(self.fc_q(x).view(-1, self.n_actions, self.n_out), dim=-1)
        elif self.algo == "QRDQN":
            action_value = self.fc_q(x).view(-1, self.n_actions, self.n_out)

        return action_value.squeeze()


class CnnPolicy(PolicyNetwork):
    def __init__(
        self,
        algo: str,
        n_actions: int,
        state_len: int = 1,
        n_out: int = -1
    ):
        """
            n_out: If given, network will be built for C51 algorithm
        """

        super().__init__()
        self.algo = algo

        # Expected input tensor shape: (B, state_len, 84, 84)
        # Input (B, 210, 160, 3) will be processed by `ProcessFrame84` wrapper -> (B, 84, 84, state_len)
        self.conv = nn.Sequential(
            nn.Conv2d(state_len, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, 512)

        if self.algo == "DQN":
            self.fc_q = nn.Linear(512, n_actions)
        else:
            self.fc_q = nn.Linear(512, n_actions * n_out)

        # action value distribution
        self.n_actions = n_actions
        self.n_out = n_out
        self.state_len = state_len

        self._init_weight()

    def forward(self, x):
        if x.dim() == 2: # When n_envs == 1 and state_len == 1
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and self.state_len == 1: # When n_envs > 1
            x = x.unsqueeze(1)
        elif x.dim() == 3 and self.state_len != 1: # When n_envs == 1
            x = x.unsqueeze(0)

        x = self.conv(x / 255.0)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        
        if self.algo == "DQN":
            action_value = self.fc_q(x)
        elif self.algo == "C51":
            action_value = F.softmax(self.fc_q(x).view(-1, self.n_actions, self.n_out), dim=-1)
        elif self.algo == "QRDQN":
            action_value = self.fc_q(x).view(-1, self.n_actions, self.n_out)

        return action_value.squeeze()


def get_policy_networks(
    algo: str,
    policy_type: str,
    state_len: int,
    n_act: int,
    n_in: Union[int, Tuple[int, int], Tuple[int, int, int]],
    n_out: int = -1,
    hidden_sizes: List[int] = None
) -> PolicyNetwork:
    if policy_type == "MlpPolicy":
        return MlpPolicy(
            algo=algo,
            n_actions=n_act,
            input_size=n_in,
            hidden_sizes=hidden_sizes,
            state_len=state_len,
            n_out=n_out
        )
    elif policy_type == "CnnPolicy":
        return CnnPolicy(
            algo=algo,
            n_actions=n_act,
            state_len=state_len,
            n_out=n_out
        )
    else:
        raise ValueError(policy_type)
