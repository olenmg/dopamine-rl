from . import *
import torch
import torch.nn as nn


class Config:
    def __init__(
        self,
        run_name: str = None,
        env_id: str = "CartPole-v1",
        random_seed: int = 42,
        use_target: bool = True,
        optim_cls: type = torch.optim.Adam,
        optim_kwargs: dict = {'lr': 0.0003},
        loss_fn: nn.modules.loss = torch.nn.SmoothL1Loss(),
        eps_start: float = 0.3,
        eps_end: float = 0.01,
        eps_decay: float = 0.9995,  
        discount_rate: float = 0.98,
        n_steps: int = 1,
        double: bool = False,
        buffer_size: int = 100000,
        batch_size: int = 128,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 2048,
        device="cpu",
        verbose=False,
    ):
        self.run_name = run_name
        self.env_id = env_id
        self.random_seed = random_seed
        self.use_target = use_target
        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs
        self.loss_fn = loss_fn
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.discount_rate = discount_rate
        self.n_steps = n_steps
        self.double = double
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.device = device
        self.verbose = verbose

    def save(self):
        pass

    def load(self):
        pass
