from copy import deepcopy
from collections import deque
from typing import Union, List

import numpy as np
import torch

from utils.config import TrainConfig, DQNConfig
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import get_env


class DQN(object):
    def __init__(
        self,
        train_config: TrainConfig,
        algo_config: DQNConfig
    ):
        assert isinstance(algo_config, DQNConfig), "Given config instance should be a DQNConfig class."

        # DQN configurations
        self.eps = algo_config.eps_start
        self.eps_decay = algo_config.eps_decay
        self.eps_end = algo_config.eps_end

        self.gamma = algo_config.discount_rate

        self.tau = algo_config.soft_update_rate

        self.learning_starts = algo_config.learning_starts
        self.train_freq = algo_config.train_freq
        self.target_update_freq = algo_config.target_update_freq

        self.memory = ReplayBuffer(size=algo_config.buffer_size)

        # Train configurations
        self.run_name = train_config.run_name
        self.env = get_env(
            env_id=train_config.env_id,
            n_envs=train_config.n_envs
        ) #TODO: state_len
        self.act_n = self.env.unwrapped.action_space[0].n

        self.n_envs = train_config.n_envs
        self.state_len = train_config.state_len

        self.batch_size = train_config.batch_size
        self.train_steps = train_config.train_step
        self.save_freq = train_config.save_freq

        self.device = torch.device(train_config.device)        
        self.verbose = train_config.verbose

        # Policy network & Training utilities
        self.pred_net = algo_config.policy_network.to(self.device)
        self.target_net = deepcopy(self.pred_net).to(self.device)
        self.target_net.eval()

        self.criterion = train_config.loss_fn
        self.optimizer = train_config.optim_cls(
            params=self.pred_net.parameters(),
            **train_config.optim_kwargs
        )

        # Others
        self.rng = np.random.default_rng(train_config.random_seed)
        self.buffer_cnt = 0

    def train(self) -> List[int]:
        episode_deque = deque(maxlen=100)
        episode_infos = []

        obs, _ = self.env.reset() # (n_envs, state_len, *)
        for step in range(self.train_steps // self.n_envs):
            action = self.predict(obs, self.eps) # (n_envs, *)

            # Take a step and store it on buffer
            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            self.add_to_buffer(obs, action, next_obs, reward, terminated, truncated)

            # Logging
            for info in infos.get("final_info", []):
                if info:
                    episode_infos.append(info["episode"])
                    episode_deque.append(info["episode"]["r"])

            # Learning if enough timestep has been gone 
            if (self.buffer_cnt >= self.learning_starts) \
                    and (self.buffer_cnt % self.train_freq == 0):
                self.update_network()

            # Periodically copies the parameter of the pred network to the target network
            if step % self.target_update_freq == 0:
                self.update_target()

            # Verbose (stdout logging)
            if self.verbose: # Logging only representative env info. (1st env.)
                status_string = f"{self.run_name:10} | STEP: {step} | REWARD: {np.mean(episode_deque):.2f} | Epsilon: {self.eps:.3f}"
                print(status_string + "\r", end="", flush=True)

            obs = next_obs

            # Update the epsilon value
            self.update_epsilon()
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
        self.buffer_cnt += 1
        done = np.any([terminated, truncated], axis=0)
        for i in range(self.n_envs):
            self.memory.add(obs[i], action[i], reward[i], next_obs[i], done[i])

    # Update online network with samples in the replay memory. 
    def update_network(self) -> None:
        # Do sampling from the buffer
        obses, actions, rewards, next_obses, dones = tuple(map(
            lambda x: torch.from_numpy(x).to(self.device),
            self.memory.sample(self.batch_size)
        ))

        # Get q-value from the target network
        with torch.no_grad():
            q_val = torch.max(self.target_net(next_obses), dim=-1).values
        y = rewards + self.gamma * ~dones * q_val # ~dones == (1 - dones)

        # Forward pass & Backward pass
        self.pred_net.train()
        self.optimizer.zero_grad()
        pred = self.pred_net(obses).gather(1, actions.unsqueeze(1)).squeeze(-1)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

    # Update the target network's weights with the online network's one. 
    def update_target(self) -> None:
        for pred_param, target_param in \
                zip(self.pred_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(
                self.tau * pred_param.data + (1.0 - self.tau) * target_param
            )
    
    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(
        self,
        obses: Union[list, np.ndarray],
        eps: float = -1.0
    ) -> np.ndarray:
        """
            obses: 
                Training stage : (n_envs, state_len, *, ) or (state_len, *, ) or (*, _)
                Inference stage: (batch_size, state_len, *, ) or (state_len, *, _), or (*, _)
            eps:
                -1.0 at inference stage
        """
        if isinstance(obses, list):
            obses = np.array(list)
        if isinstance(obses, np.ndarray):
            obses = torch.from_numpy(obses)
        # obses = obses.squeeze() # Squeezed when n_envs == 1 or state_len == 1

        # Epsilon-greedy
        if self.rng.random() >= eps:
            self.pred_net.eval()
            with torch.no_grad():
                action = torch.argmax(
                    self.pred_net(obses.to(self.device)), dim=-1
                ).cpu().numpy()
        else:
            action = self.rng.choice(self.act_n, size=(self.n_envs, ))

        return action
    
    # Update epsilon over training process.
    def update_epsilon(self) -> None:
        self.eps = max(
            self.eps * self.eps_decay,
            self.eps_end
        )
