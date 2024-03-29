import os
import time
import json
from copy import deepcopy
from importlib import import_module
from collections import deque
from typing import List, Union

import numpy as np
import torch
from tqdm import tqdm

from utils.wrappers import get_env
from utils.config import TrainConfig, DQNConfig, C51Config, QRConfig
from utils.policy_networks import get_policy_networks


class RLAlgorithm(object):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: Union[DQNConfig, C51Config, QRConfig],
        render: bool = False
    ):
        # Train configurations
        self.run_name = train_config.run_name
        self.env = get_env(train_config, render=render, render_mode="rgb_array")
        self.n_act = self.env.unwrapped.action_space.n if train_config.n_envs == 1 else self.env.unwrapped.action_space[0].n

        self.n_envs = train_config.n_envs
        self.state_len = train_config.state_len
        self.reward_clipping = train_config.reward_clipping

        self.batch_size = train_config.batch_size
        self.train_steps = train_config.train_step
        self.save_freq = train_config.save_freq
        self.logging_freq = train_config.logging_freq

        self.device = torch.device(train_config.device)        
        self.verbose = train_config.verbose

        self.loss_cls, self.loss_kwargs = train_config.loss_cls, train_config.loss_kwargs
        self.optim_cls, self.optim_kwargs = train_config.optim_cls, train_config.optim_kwargs

        # For logging & Save
        self.save_path = os.path.join("results", train_config.run_name)
        if not render:
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "train_cfg.json"), "w") as f:
                f.write(json.dumps(train_config.__dict__, indent=4))
            with open(os.path.join(self.save_path, "algo_cfg.json"), "w") as f:
                f.write(json.dumps(algo_config.__dict__, indent=4))

        # Others
        self.rng = np.random.default_rng(train_config.random_seed)

    # Update online network
    def update_network(self) -> None:
        raise NotImplementedError

    # Return desired action(s)
    def predict(
        self,
        obses: Union[list, np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError

    def save_model(self) -> None:
        raise NotImplementedError

    def load_model(self) -> None:
        raise NotImplementedError


class ValueIterationAlgorithm(RLAlgorithm):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: Union[DQNConfig, C51Config, QRConfig],
        render: bool = False
    ):
        super().__init__(train_config=train_config, algo_config=algo_config, render=render)

        # Policy networks
        self.pred_net = get_policy_networks(
            algo=train_config.algo,
            state_len=self.state_len,
            n_act=self.n_act,
            n_in=self.env.observation_space.shape[-1],
            n_out=getattr(algo_config, "n_out", -1),
            **algo_config.policy_kwargs
        ).to(self.device)
        self.target_net = deepcopy(self.pred_net).to(self.device)
        self.target_net.eval()

        self.criterion = getattr(
            import_module("utils.loss") if self.loss_cls in dir(import_module("utils.loss")) \
                else import_module("torch.nn"), self.loss_cls
        )(**self.loss_kwargs)
        self.optimizer = getattr(
            import_module("torch.optim"),
            self.optim_cls
        )(params=self.pred_net.parameters(), **self.optim_kwargs)
        self.eps_scheduler = getattr(
            import_module("utils.eps_scheduler"),
            algo_config.eps_cls
        )(**algo_config.eps_kwargs)
        self.eps = self.eps_scheduler.get_eps()

    def train(self) -> List[int]:
        episode_deque = deque(maxlen=100)
        episode_infos = []
        start_time = time.time()

        best_reward = float('-inf')

        obs, _ = self.env.reset() # (n_envs, state_len, *)
        for step in tqdm(range(self.train_steps // self.n_envs)):
            action = self.predict(obs, self.eps) # (n_envs, *)

            # Take a step and store it on buffer
            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            self.add_to_buffer(obs, action, next_obs, reward, terminated, truncated)

            # Logging
            if self.n_envs == 1: # Single environment
                if terminated or truncated:
                    episode_infos.append(infos["episode"])
                    episode_deque.append(infos["episode"]["r"])
                    next_obs, _ = self.env.reset()
            else: # Vectorized environment
                for info in infos.get("final_info", []):
                    if info:
                        episode_infos.append(info["episode"])
                        episode_deque.append(info["episode"]["r"])

            # Learning if enough timestep has been gone 
            if (self.buffer_cnt >= self.learning_starts) \
                    and (self.buffer_cnt % self.train_freq == 0):
                self.update_network() # Train using replay buffer

            # Periodically copies the parameter of the pred network to the target network
            if step % self.target_update_freq == 0:
                self.update_target()
            obs = next_obs
            self.update_epsilon() # Epsilon-greedy
            
            # Logging
            if episode_deque:
                # Verbose (stdout logging)
                if (step % self.logging_freq == 0):
                    used_time = time.time() - start_time
                    print(f"Step: {step} |",
                        f"100-mean reward: {np.mean(episode_deque):.2f} |",
                        f"Latest reward: {episode_deque[-1]:.2f} |",
                        f"Epsilon: {self.eps:.3f} |",
                        f"Used time: {used_time:.3f}"
                    )

                # Save the model
                if self.verbose and (step % self.save_freq == 0):
                    self.save_model()
                    # Save the best model (roughly best)
                    if episode_deque and episode_deque[-1] > best_reward:
                        self.save_model("best_pred_net.pt", "best_target_net.pt")
                        best_reward = episode_deque[-1]
        self.env.close()

        return episode_infos

    def add_to_buffer(
        self,
        obs: np.ndarray, # float, (n_envs, state_len, *)
        action: np.ndarray, # int, (n_envs, *)
        next_obs: np.ndarray, # float, (n_envs, state_len, *)
        reward: np.ndarray, # int, (n_envs, *)
        terminated: np.ndarray, # bool, (n_envs, *)
        truncated: np.ndarray # bool, (n_envs, *)
    ) -> None:
        if self.n_envs == 1: # Single environment
            done = terminated or truncated
            self.memory.add(obs, action, reward, next_obs, done)
            self.buffer_cnt += 1
        else: # Vectorized environment
            done = np.any([terminated, truncated], axis=0)
            for i in range(self.n_envs):
                self.memory.add(obs[i], action[i], reward[i], next_obs[i], done[i])
            self.buffer_cnt += self.n_envs

    # Update the target network's weights with the online network's one. 
    def update_target(self) -> None:
        for pred_param, target_param in \
                zip(self.pred_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(
                self.tau * pred_param.data + (1.0 - self.tau) * target_param
            )

    # Update epsilon over training process.
    def update_epsilon(self) -> None:
        self.eps = self.eps_scheduler.step()

    # Save model
    def save_model(
        self,
        pred_net_fname: str = "pred_net.pt",
        target_net_fname: str = "target_net.pt"
    ) -> None:
        torch.save(self.pred_net.state_dict(), os.path.join(self.save_path, pred_net_fname))
        torch.save(self.target_net.state_dict(), os.path.join(self.save_path, target_net_fname))

    # Save model
    def load_model(
        self,
        pred_net_fname: str = "pred_net.pt",
        target_net_fname: str = "target_net.pt"
    ) -> None:
        if pred_net_fname == "pred_net.pt" and target_net_fname == "target_net.pt":
            self.pred_net.load_state_dict(torch.load(
                os.path.join(self.save_path, pred_net_fname),
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.target_net.load_state_dict(torch.load(
                os.path.join(self.save_path, target_net_fname),
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
        else:
            self.pred_net.load_state_dict(torch.load(
                pred_net_fname,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.target_net.load_state_dict(torch.load(
                target_net_fname,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            
