from copy import deepcopy
from typing import Union

import numpy as np
import torch

from rl_algo.algorithm import ValueIterationAlgorithm
from utils.replay_buffer import ReplayBuffer
from utils.config import TrainConfig, QRConfig
from utils.policy_networks import get_policy_networks


class QRDQN(ValueIterationAlgorithm):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: QRConfig
    ):
        super().__init__(train_config=train_config, algo_config=algo_config)
        assert isinstance(algo_config, QRConfig), "Given config instance should be a QRConfig class."

    # Update online network with samples in the replay memory. 
    def update_network(self) -> None:
        pass

    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(
        self,
        obses: Union[list, np.ndarray],
        eps: float = -1.0
    ) -> np.ndarray:
        pass
