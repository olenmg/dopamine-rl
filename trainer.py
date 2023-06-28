from copy import deepcopy
from typing import Union, List

import numpy as np
import gymnasium as gym
import torch

from utils.replay_buffer import ReplayBuffer
from model import SimpleMLP


class Trainer(object):
    def __init__(
        self,
        config
    ):
        self.config = config
        self.env = gym.make(config.env_id)

        self.epsilon = config.eps_start
        self.gamma = config.discount_rate
        self.batch_size = config.batch_size
        self.n_steps = config.n_steps
        self.use_target = config.use_target

        self.rng = np.random.default_rng(config.random_seed)

        self.device = torch.device(config.device)

        self.obs_n = self.env.observation_space.shape
        self.act_n = self.env.action_space.n
        
        self.mlp_pred = SimpleMLP(self.obs_n[0], self.act_n, hidden_sizes=[128, 128]).to(self.device)
        self.mlp_target = deepcopy(self.mlp_pred).to(self.device)
        self.mlp_target.eval()

        self.criterion = config.loss_fn
        self.optimizer = config.optim_cls(
            params=self.mlp_pred.parameters(),
            **config.optim_kwargs
        )

        self.memory = ReplayBuffer(size=config.buffer_size)

    def train(
        self,
        num_train_steps: int = -1,
        num_train_eps: int = -1
    ) -> List[int]:
        assert (num_train_steps != -1) or (num_train_eps != -1), "Specify either the number of training steps or episodes."
        assert (num_train_steps == -1) or (num_train_eps == -1), "Only one of the training steps and episodes is required."
        #TODO operation for num_train_eps

        i = 0
        episode_rewards = []
        episode_reward = 0

        obs, _ = self.env.reset()
        while i < num_train_steps:
            i += 1

            # Epsilon-greedy
            if self.rng.random() > self.epsilon:
                action = self.predict(obs)
            else:
                action = self.rng.choice(self.act_n)

            # Take a step and store it on buffer
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.add(obs, action, reward, next_obs, done)

            # Learning if enough timestep has been gone 
            if (i >= self.config.learning_starts) and (i % self.config.train_freq == 0):
                self.update_network()

            # Periodically copies the parameter of the pred network to the target network
            if self.use_target and ((i % self.config.target_update_freq) == 0):
                self.update_target()

            # Verbose (stdout logging)
            if self.config.verbose:
                status_string = f"{self.config.run_name:10} | STEP: {i} | REWARD: {episode_reward}"
                print(status_string + "\r", end="", flush=True)

            # Update the total episode reward & Check if the episode has been done
            episode_reward += reward
            if done:
                obs, _ = self.env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0
            else:
                obs = next_obs

            # Update the epsilon value
            self.update_epsilon()

        return episode_rewards

    # Update online network with samples in the replay memory. 
    def update_network(self) -> None:
        # Do sampling from the buffer
        obses, actions, rewards, next_obses, dones = tuple(map(
            lambda x: torch.from_numpy(x).to(self.device),
            self.memory.sample(self.batch_size)
        ))

        # Get q-value from the target network
        with torch.no_grad():
            q_val = torch.max(self.mlp_target(next_obses), dim=-1).values
        y = rewards + self.gamma * ~dones * q_val # ~dones == (1 - dones)

        # Forward pass & Backward pass
        self.mlp_pred.train()
        self.optimizer.zero_grad()
        pred = self.mlp_pred(obses).gather(1, actions.unsqueeze(1)).squeeze(-1)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

    # Update the target network's weights with the online network's one. 
    def update_target(self) -> None:
        self.mlp_target.load_state_dict(self.mlp_pred.state_dict())
    
    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(self, obses: Union[list, np.ndarray]) -> int:
        if isinstance(obses, list):
            obses = np.array(list)
        if isinstance(obses, np.ndarray):
            obses = torch.from_numpy(obses)

        self.mlp_pred.eval()
        with torch.no_grad():
            action = torch.argmax(self.mlp_pred(obses.to(self.device))).cpu().item()
        
        return action
    
    # Update epsilon over training process.
    def update_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon * self.config.eps_decay,
            self.config.eps_end
        )
