from rl_algo.dqn import DQN
from rl_algo.c51 import C51
from rl_algo.qrdqn import QRDQN
from rl_algo.mgdqn import MGDQN
from rl_algo.mgc51 import MGC51

ALGO = {
    "DQN": DQN,
    "C51": C51,
    "QRDQN": QRDQN,
    "MGDQN": MGDQN,
    "MGC51": MGC51
}

__all__ = [
    "ALGO",
    "DQN",
    "C51",
    "QRDQN",
    "MGDQN",
    "MGC51"
]
