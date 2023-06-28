import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64,]):
        super().__init__()
        
        self.inp_layer = nn.Linear(input_size, hidden_sizes[0])

        self.linears = nn.ModuleList(
            [
                nn.Linear(in_shape, out_shape) for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self.out_layer = nn.Linear(hidden_sizes[-1], output_size)
    
    
    def forward(self, x):
        x = F.relu(self.inp_layer(x))

        for layer in self.linears:
            x = F.relu(layer(x))
        
        return self.out_layer(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64,]):
        super().__init__()
        
        self.inp_layer = nn.Linear(input_size, hidden_sizes[0])

        self.linears = nn.ModuleList(
            [
                nn.Linear(in_shape, out_shape) for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self.out_layer = nn.Linear(hidden_sizes[-1], output_size)
    
    
    def forward(self, x):
        x = F.relu(self.inp_layer(x))

        for layer in self.linears:
            x = F.relu(layer(x))
        
        return self.out_layer(x)
