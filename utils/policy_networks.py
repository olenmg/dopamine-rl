from typing import List

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


class MLPNet(PolicyNetwork):
    def __init__(
        self,
        n_actions: int,
        input_size: int,
        hidden_sizes: List[int] = [64, ],
        state_len: int = 1,
        n_atom: int = -1
    ):
        """
            n_atom: If given, network will be built for C51 algorithm
        """

        super().__init__()        
        self.in_layer = nn.Linear(input_size * state_len, hidden_sizes[0])
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_shape, out_shape) for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        if n_atom == -1:
            self.fc_q = nn.Linear(hidden_sizes[-1], n_actions)
        else:
            self.fc_q = nn.Linear(hidden_sizes[-1], n_actions * n_atom)

        self.n_actions = n_actions
        self.n_atom = n_atom

        self._init_weight()
    
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for layer in self.linears:
            x = F.relu(layer(x))

        if self.n_atom == -1:
            action_value = self.fc_q(x)
        else:
            action_value = F.softmax(self.fc_q(x).view(-1, self.n_actions, self.n_atom), dim=-1)
        return action_value


class ConvNet(PolicyNetwork):
    def __init__(
        self,
        n_actions: int,
        state_len: int = 1,
        n_atom: int = -1
    ):
        """
            n_atom: If given, network will be built for C51 algorithm
        """

        super().__init__()
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

        if n_atom == -1:
            self.fc_q = nn.Linear(512, n_actions)
        else:
            self.fc_q = nn.Linear(512, n_actions * n_atom)

        # action value distribution
        self.n_actions = n_actions
        self.n_atom = n_atom

        self._init_weight()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x / 255.0) #TODO: Check
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        
        if self.n_atom == -1:
            action_value = self.fc_q(x)
        else:
            action_value = F.softmax(self.fc_q(x).view(-1, self.n_actions, self.n_atom), dim=-1)

        return action_value
