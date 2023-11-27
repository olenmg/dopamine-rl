import random
from random import randrange
from typing import Tuple, List, Union

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BallCrashing(gym.Env):
    """ Ball crashing came """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        num_good: int = 8,
        num_bad: int = 0,
        reward_range: Union[Tuple, List] = (-5, 5),
        grid_size: int = 8,
        day_len: int = 300,
        all_day: bool = True,
        render_mode: str = None
    ):
        super().__init__()
        
        assert len(reward_range) == 2, "A length of `reward_range` should be 2."
        assert reward_range[1] > 0 and reward_range[0] < 0, "`reward_range` should cover the range from negative to positive."
        print("[Ball Crashing] Ready for playing!")

        self.grid_size = grid_size
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(grid_size * grid_size, ), dtype=np.float32
        ) # Ball: 0, Other: reward

        self.num_good = num_good
        self.num_bad = num_bad
        self.reward_range = reward_range
        self.day_len = day_len

        self.ball_types = self.make_ball_types(
            self.num_good,
            self.num_bad,
            self.reward_range
        ) # {color: reward}

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state        
        self.start_time = randrange(20)
        self.end_time = randrange(self.day_len - 20, self.day_len)
        self.cur_time = self.start_time
        self.episodic_reward = 0 # for logging 

        # self.crd = self.generate_coordinate(gen_agent=True)
        # self.balls = dict() # {coordinates: reward}
        # for _ in range(self.num_good):
        #     self.make_new_ball(ball_type='good')
        # for _ in range(self.num_bad):
        #     self.make_new_ball(ball_type='bad')
        self.crd = (0, 0)
        self.balls = {(3, 3): 0.625, (2, 0): 3.125, (0, 2): 5.0, (7, 3): 4.375, (4, 5): 0.625, (0, 4): 0.625, (6, 6): 1.875, (0, 7): 2.5}

        info = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "agent_pos": self.crd,
            "balls": self.balls
        }

        return self.get_grid_state(), info

    def step(self, action):
        self.cur_time += 1

        prev_crd = self.crd
        if action == 0: # No-op
            pass  
        elif action == 1: # Up
            self.crd = (max(self.crd[0] - 1, 0), self.crd[1])
        elif action == 2: # Down
            self.crd = (min(self.crd[0] + 1, self.grid_size - 1), self.crd[1])
        elif action == 3: # Left
            self.crd = (self.crd[0], max(self.crd[1] - 1, 0))
        elif action == 4: # Right
            self.crd = (self.crd[0], min(self.crd[1] + 1, self.grid_size - 1))

        # Reward
        # reward = 0
        reward = -1 if (self.crd == prev_crd) else -0.001 # Penalty
        # reward = 0 if action == 0 else -0.01 # Moving penalty
        for ex_crd in list(self.balls.keys()):
            if (ex_crd[0] == self.crd[0]) and (ex_crd[1] == self.crd[1]):
                reward += self.balls.pop(ex_crd)
                self.make_new_ball('good' if reward > 0 else 'bad')

        self.episodic_reward += reward
        done = (self.cur_time == self.end_time)

        # Info
        info = {
            "cur_time": self.cur_time,
            "reward": reward
        }

        return self.get_grid_state(), reward, done, False, info

    def render(self):
        CELL_SIZE = 30
        BOARD_SIZE = self.grid_size * CELL_SIZE
        # Create a blank image
        img = np.ones(
            (BOARD_SIZE, BOARD_SIZE, 3),
            dtype=np.uint8
        ) * 255

        # Draw the grid lines
        for x in range(0, BOARD_SIZE, CELL_SIZE):
            cv2.line(img, (x, 0), (x, BOARD_SIZE), color=(0, 0, 0), thickness=1)
        for y in range(0, BOARD_SIZE, CELL_SIZE):
            cv2.line(img, (0, y), (BOARD_SIZE, y), color=(0, 0, 0), thickness=1)

        # Add balls
        for (x, y), val in self.balls.items():
            color = (255, 0, 0) if val > 0 else (0, 0, 255)
            position = (y * CELL_SIZE + 10, x * CELL_SIZE + 15)
            cv2.putText(img, str(val), position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        # Agent
        cv2.putText(
            img, 'A', (self.crd[1] * CELL_SIZE + 10, self.crd[0] * CELL_SIZE + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        return img

    def get_grid_state(self):
        state = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * -1
        for (y, x), val in self.balls.items():
            state[y, x] = val
        state[self.crd[0], self.crd[1]] = 0
        return state

    def make_new_ball(self, ball_type: str) -> Tuple[Tuple[int, int], str]:
        assert len(self.balls) <= self.num_good + self.num_bad, "Too many balls."

        coordinate = self.generate_coordinate(gen_agent=False)
        reward = random.choice(self.ball_types[ball_type])
        self.balls.update({coordinate: reward})

        return (coordinate, reward)

    def generate_coordinate(
        self,
        gen_agent: bool = False
    ) -> Tuple[int, int]:
        def validate_coordinate(crd, existing_crds):
            for ex_crd in existing_crds:
                if (abs(ex_crd[0] - crd[0]) <= 1) and (abs(ex_crd[1] - crd[1]) <= 1):
                    return False
            return True
        
        new_crd = (randrange(0, self.grid_size), randrange(0, self.grid_size))
        if not gen_agent:        
            existing_crds = list(self.balls.keys()) + [self.crd]
            while not validate_coordinate(new_crd, existing_crds):
                new_crd = (randrange(0, self.grid_size), randrange(0, self.grid_size))
        return new_crd

    def make_ball_types(
        self,
        num_good: int,
        num_bad: int,
        reward_range: Tuple[int, int]
    ) -> List[Tuple[str, float]]:
        """ Determine the color and reward of the balls to be used. """
        candidates = {
            'good': np.linspace(0, reward_range[-1], num_good + 1)[1:],
            'bad': np.linspace(reward_range[0], 0, num_bad + 1)[:-1]
        }
        return candidates


# Debug
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation

    def get_render_frames(env, n_step=600):
        total_reward = 0
        done_counter = 0
        frames = []

        obs, _ = env.reset()
        for _ in range(n_step):
            # Render into buffer. 
            frames.append(env.render())
            action = random.randint(0, 5)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                done_counter += 1
                obs, _ = env.reset()
            else:
                obs = next_obs

            if done_counter == 2:
                break

        env.close()
        print(f"Total Reward: {total_reward:.2f}")
        return frames

    def display_frames_as_gif(frames, fname="result.gif"):
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])
            
        ani = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        ani.save(fname, writer='pillow', fps=30)


    env = BallCrashing()
    frames = get_render_frames(
        env=env,
        n_step=2000
    )

    display_frames_as_gif(
        frames=frames,
        fname="video.gif"
    )
