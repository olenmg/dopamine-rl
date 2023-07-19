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


class MLPNet(PolicyNetwork):
    def __init__(self, input_size, output_size, hidden_sizes=[64,]):
        super().__init__()        
        self.in_layer = nn.Linear(input_size, hidden_sizes[0])
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_shape, out_shape) for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self.out_layer = nn.Linear(hidden_sizes[-1], output_size)
        self._init_weight()
    
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for layer in self.linears:
            x = F.relu(layer(x))
        return self.out_layer(x)


class ConvNet(PolicyNetwork):
    def __init__(
        self,
        n_actions: int,
        n_atom: int,
        state_len: int = 1
    ):
        super().__init__()
        # Expected input tensor shape: (B, 84, 84, state_len)
        # Input (B, 210, 160, 3) will be processed by `ProcessFrame84` wrapper -> (B, 84, 84, state_len)

        self.conv = nn.Sequential(
            nn.Conv2d(state_len, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 16 * state_len, 512)
        
        # action value distribution
        self.fc_q = nn.Linear(512, n_actions * n_atom)

        self.n_actions = n_actions
        self.n_atom = n_atom

        self._init_weight()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x / 255.0) #TODO: Check
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x))
        
        # Output of C51 is PMFs of action value distribution
        action_value = F.softmax(self.fc_q(x).view(batch_size, self.n_actions, self.n_atom), dim=2)

        return action_value

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))
