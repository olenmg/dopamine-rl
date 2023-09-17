import os
import json
import argparse

from rl_algo import DQN, C51, QRDQN, MGDQN
from utils.config import TrainConfig, DQNConfig, C51Config, QRConfig, MGDQNConfig
from utils.plot import plot_train_result
from render import render

ALGO_CONFIG = {
    'DQN': DQNConfig, 'C51': C51Config, 'QRDQN': QRConfig, 'MGDQN': MGDQNConfig
}
ALGO = {
    'DQN': DQN, 'C51': C51, 'QRDQN': QRDQN, 'MGDQN': MGDQN
}

ENVS = [
    "ALE/VideoPinball-v5", "ALE/Boxing-v5", "ALE/Breakout-v5", "ALE/StarGunner-v5", 
    "ALE/Robotank-v5", "ALE/Atlantis-v5", "ALE/CrazyClimber-v5", "ALE/Gopher-v5",
    "ALE/DemonAttack-v5", "ALE/NameThisGame-v5", "ALE/Krull-v5", "ALE/Assault-v5",
    "ALE/RoadRunner-v5", "ALE/Kangaroo-v5", "ALE/Jamesbond-v5", "ALE/Tennis-v5",
    "ALE/Pong-v5", "ALE/SpaceInvaders-v5", "ALE/BeamRider-v5", "ALE/Tutankham-v5"
    "ALE/KungFuMaster-v5", "ALE/Freeway-v5", "ALE/TimePilot-v5", "ALE/Enduro-v5",
    "ALE/FishingDerby-v5", "ALE/UpNDown-v5", "ALE/IceHockey-v5", "ALE/Qbert-v5",
    "ALE/Hero-v5", "ALE/Asterix-v5", "ALE/BattleZone-v5", "ALE/WizardOfWor-v5",
    "ALE/ChopperCommand-v5", "ALE/Centipede-v5", "ALE/BankHeist-v5", "ALE/Riverraid-v5",
    "ALE/Zaxxon-v5", "ALE/Amidar-v5", "ALE/Alien-v5", "ALE/Venture-v5",
    "ALE/Seaquest-v5", "ALE/DoubleDunk-v5", "ALE/Bowling-v5", "ALE/MsPacman-v5",
    "ALE/Asteroids-v5", "ALE/Frostbite-v5", "ALE/Gravitar-v5", "ALE/PrivateEye-v5", 
    "ALE/MontezumaRevenge-v5"
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument(
        '--algo', type=str,
        help="Algorithm"
    )
    args = parser.parse_args()

    algo, algo_config = ALGO[args.algo], ALGO_CONFIG[args.algo]
    cfg_name = f"{args.algo.lower()}_atari.json"

    for env_id in ENVS:
        with open(os.path.join("configs/train_configs", cfg_name), "r") as f:
            train_config = TrainConfig(**json.load(f))
        with open(os.path.join("configs/algo_configs", cfg_name), "r") as f:
            algo_config = ALGO_CONFIG[train_config.algo](**json.load(f))
        train_config.run_name = args.log_path = f"{args.algo}-{env_id[4:-3]}"
        train_config.env_id = env_id

        model = ALGO[train_config.algo](
            train_config=train_config,
            algo_config=algo_config
        )
        result = model.train()
        plot_train_result(
            result=result,
            label=train_config.run_name,
            save_path=os.path.join("results", train_config.run_name, f"{train_config.run_name}.png"),
            alpha=0.9
        )

        args.use_best = True
        render(args)
