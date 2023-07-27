import torch
import torch.nn as nn

from .policy_networks import PolicyNetwork


class TrainConfig:
    def __init__(
        self,
        run_name: str = None,
        env_id: str = "CartPole-v1",
        n_envs: int = 4,
        state_len: int = 1,
        frame_skip: int = 1,
        random_seed: int = 42,
        loss_cls: str = "SmoothL1Loss",
        loss_kwargs: dict = {},
        optim_cls: str = "Adam",
        optim_kwargs: dict = {'lr': 0.0003},
        batch_size: int = 128,
        train_step: int = int(1e+6),
        save_freq: int = -1,
        logging_freq: int = 10000,
        device: str = "auto",
        verbose: bool = False
    ):
        assert device in ["cpu", "cuda", "mps", "auto"], "Device should be one of 'cpu', 'cuda', 'mps'."
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is now unavailable."
        elif device == "mps":
            assert torch.backends.mps.is_available(), "mps is now unavailable."
        elif device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.run_name = run_name
        self.env_id = env_id
        self.n_envs = n_envs
        self.state_len = state_len # Sequential images to define state
        self.frame_skip = frame_skip
        self.random_seed = random_seed
        self.loss_cls = loss_cls
        self.loss_kwargs = loss_kwargs
        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs
        self.batch_size = batch_size
        self.train_step = train_step
        self.save_freq = save_freq
        self.logging_freq = logging_freq
        self.device = device
        self.verbose = verbose


class DQNConfig:
    def __init__(
        self,
        policy_kwargs: dict,
        eps_start: float = 0.3,
        eps_end: float = 0.01,
        eps_decay: float = 0.9995,  
        discount_rate: float = 0.98,
        soft_update_rate: float = 1.0,
        buffer_size: int = 100000,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 2048,
        n_atom: int = -1 # Not used
    ):
        self.policy_kwargs = policy_kwargs
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.discount_rate = discount_rate
        self.soft_update_rate = soft_update_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.n_atom = -1


class C51Config(DQNConfig):
    def __init__(
        self,
        policy_kwargs: dict,
        v_min: float,
        v_max: float,
        n_atom: int = 51,
        eps_start: float = 0.3,
        eps_end: float = 0.01,
        eps_decay: float = 0.9995,  
        discount_rate: float = 0.98,
        soft_update_rate: float = 1.0,
        buffer_size: int = 100000,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 512,
    ):
        super().__init__(
            policy_kwargs=policy_kwargs,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            discount_rate=discount_rate,
            soft_update_rate=soft_update_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            target_update_freq=target_update_freq,
        )
        self.v_min = v_min
        self.v_max = v_max
        self.n_atom = n_atom


class QRConfig: #TODO
    def __init__(
        self,
        policy_kwargs: dict,
        eps_start: float = 0.3,
        eps_end: float = 0.01,
        eps_decay: float = 0.9995,  
        discount_rate: float = 0.98,
        soft_update_rate: float = 1.0,
        buffer_size: int = 100000,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 2048
    ):
        self.policy_kwargs = policy_kwargs
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.discount_rate = discount_rate
        self.soft_update_rate = soft_update_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq