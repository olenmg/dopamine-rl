# The MIT License

# Copyright (c) 2019 Antonin Raffin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Excerpted from DLR-RM/stable-baselines3: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/monitor.py

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spec
from gymnasium.spaces import Box
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import AtariPreprocessing, FrameStack

from rl_env import CUSTOM_ENVS


class MouseVRPreprocessing(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """ Mouse VR preprocessing wrapper.

    This class refers to gymnasium.wrappers.atari_preprocessing.py

    Specifically, the following preprocess stages applies to the mouse VR environment:

    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    - Frame skipping: The number of frames skipped between steps, 4 by default
    - Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default
    - Grayscale observation: If the observation is colour or greyscale, by default, greyscale.
    - Scale observation: If to scale the observation between [0, 1) or [0, 255), by default, not scaled.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        grayscale_obs: bool = True,
        scale_obs: bool = False,
    ):
        """Wrapper for Atari 2600 preprocessing.

        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
            screen_size (int): resize Atari frame
            grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
                is returned.
            scale_obs (bool): if True, then observation normalized in range [0,1) is returned. It also limits memory
                optimization benefits of FrameStack Wrapper.

        Raises:
            DependencyNotInstalled: opencv-python package not installed
            ValueError: Disable frame-skipping in the original env
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            grayscale_obs=grayscale_obs,
            scale_obs=scale_obs,
        )
        gym.Wrapper.__init__(self, env)

        if cv2 is None:
            raise gym.error.DependencyNotInstalled(
                "opencv-python package not installed, run `pip install gymnasium[other]` to get dependencies for atari"
            )
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            if (
                env.spec is not None
                and "NoFrameskip" not in env.spec.id
                and getattr(env.unwrapped, "_frameskip", None) != 1
            ):
                raise ValueError(
                    "Disable frame-skipping in the original env. Otherwise, more than one "
                    "frame-skip will happen as through this wrapper"
                )
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale_obs = grayscale_obs
        self.scale_obs = scale_obs

        assert isinstance(env.observation_space, Box)
        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs:
            _shape = _shape[:-1]  # Remove channel axis     
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

        if grayscale_obs:
            env.cvt_screen_gray_scale()

    def step(self, action):
        """Applies the preprocessing for an :meth:`env.step`."""
        obs, total_reward, terminated, truncated, info = None, 0.0, False, False, {}

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        obs = cv2.resize(
            obs,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment using preprocessing."""
        # NoopReset
        obs = None
        _, reset_info = self.env.reset(**kwargs)

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )

        for _ in range(noops):
            obs, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                _, reset_info = self.env.reset(**kwargs)

        obs = cv2.resize(
            obs,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        return obs, reset_info


class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env=env)
        self.t_start = time.time()
        self.results_writer = None
        if filename is not None:
            env_id = env.spec.id if env.spec is not None else None
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=reset_keywords + info_keywords,
                override_existing=override_existing,
            )

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards: List[float] = []
        self.needs_reset = True
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: Dict[str, Any] = {}

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super().close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times


class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class

    :param filename: the location to save a log file. When it does not end in
        the string ``"monitor.csv"``, this suffix will be appended to it
    :param header: the header dictionary object of the saved csv
    :param extra_keys: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    def __init__(
        self,
        filename: str = "",
        header: Optional[Dict[str, Union[float, str]]] = None,
        extra_keys: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        filename = os.path.realpath(filename)
        # Create (if any) missing filename directories
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Append mode when not overridding existing file
        mode = "w" if override_existing else "a"
        # Prevent newline issue on Windows, see GH issue #692
        self.file_handler = open(filename, f"{mode}t", newline="\n")
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t", *extra_keys))
        if override_existing:
            self.file_handler.write(f"#{json.dumps(header)}\n")
            self.logger.writeheader()

        self.file_handler.flush()

    def write_row(self, epinfo: Dict[str, float]) -> None:
        """
        Write row of monitor data to csv log file.

        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()


def get_env(train_config, render=False, **kwargs) -> gym.Env:
    # Wrapping
    def wrap_(i):
        log_path = os.path.join("results", train_config.run_name, f"{i}.monitor.csv")
        # Make the environmemnt
        if "ALE" in train_config.env_id: # Atari games
            env = gym.make(train_config.env_id, **kwargs, frameskip=train_config.frame_skip)
            if not render:
                env = Monitor(env, log_path)
            env = AtariPreprocessing(env)
        elif train_config.env_id in CUSTOM_ENVS: # Custom envs
            env = CUSTOM_ENVS[train_config.env_id](**kwargs)
            if not render:
                env = Monitor(env, log_path)
            env = MouseVRPreprocessing(env)
        else: # Other envs
            if train_config.frame_skip == 1:
                kwargs["frameskip"] = train_config.frame_skip
            env = gym.make(train_config.env_id, **kwargs)
            if not render:
                env = Monitor(env, log_path)

        if train_config.state_len > 1:
            env = FrameStack(env, num_stack=train_config.state_len)

        return env
    env = gym.vector.AsyncVectorEnv([lambda x=i: wrap_(x) for i in range(train_config.n_envs)])
    return env 


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    from utils.config import TrainConfig

    train_config = TrainConfig(
        run_name="test",
        env_id="CartPole-v1",
        n_envs=4,
        state_len=1, 
        frame_skip=1,
        random_seed=42,
        optim_cls=torch.optim.Adam,
        optim_kwargs={'lr': 1.1e-3},
        loss_fn=nn.SmoothL1Loss(),
        batch_size=128,
        train_step=int(1e5),
        save_freq=-1,
        device="cpu",
        verbose=True
    )
    env = get_env(train_config)
    s, _ = env.reset()
