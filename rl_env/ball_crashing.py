import random
from random import randrange
from typing import Tuple, List, Union

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# BGR Colors for CV2
COLORS = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'cyan': (255, 255, 0),
    'silver': (192, 192, 192),
    'maroon': (0, 0, 128),
    'purple': (128, 0, 128),
    'navy': (128, 0, 0),
    'orange': (0, 165, 255),
    'pink': (203, 192, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'magenta': (255, 0, 255),
    'gray': (128, 128, 128),
    'olive': (0, 128, 128),
    'teal': (128, 128, 0),
    'brown': (42, 42, 165),
    'lime': (0, 255, 0),
    'skyblue': (235, 206, 135)
}

class Whiteboard(object):
    def __init__(self, day_len, radius=5, size=(84, 84), bg_color=(255, 255, 255)):
        self.day_len = day_len
        self.radius = radius
        self.size = size
        self.bg_color = bg_color
        self.agent_color = (0, 0, 0) # Black
        self.agent_position = None

        self.circles = dict()  # List to keep track of circles
        self.start_time = None
        self.end_time = None
        self.time = None

    def init_board(self, start_time, end_time, agent_position):
        self._board = np.full((self.size[0], self.size[1], 3), self.bg_color, dtype=np.uint8)
        self.circles = dict()
        self.start_time = start_time
        self.end_time = end_time
        self.time = start_time

        self.agent_position = agent_position
        cv2.circle(self._board, self.agent_position, self.radius, self.agent_color, -1)

    def add_circle(self, position, color):
        """ Add a circle to the whiteboard at a specified position. """
        self.circles[position] = color
        cv2.circle(self._board, position, self.radius, color, -1)

    def remove_circle(self, position):
        """ Remove a circle from the whiteboard at a specified position. """
        self.circles.pop(position)
        cv2.circle(self._board, position, self.radius, self.bg_color, -1)

    def move_agent(self, new_position):
        """ Move the agent to a new position. """
        self.time += 1

        cv2.circle(self._board, self.agent_position, self.radius, self.bg_color, -1)
        self.agent_position = new_position

        for (position, color) in list(self.circles.items()):
            cv2.circle(self._board, position, self.radius, color, -1)
        cv2.circle(self._board, self.agent_position, self.radius, self.agent_color, -1)

    def get_state(self):
        return self.saturate_dark(
            self._board, 
            self.get_brightness()
        ).transpose(2, 0, 1)

    def get_frame_for_render(self):
        frame = self.saturate_dark(
            self._board,
            self.get_brightness()
        ) # Agent's view
        frame = np.hstack((
            frame, 
            np.full((84, 2, 3), self.bg_color, dtype=np.uint8),
            self._board
        )) # Plot with omniscient view (84, 170, 3)

        # Scale-up
        frame = cv2.resize(frame, (510, 192), interpolation=cv2.INTER_LINEAR)

        # Put the title
        frame = np.vstack((
            np.full((20, 510, 3), self.bg_color, dtype=np.uint8),
            frame
        ))
        frame[:, 252:258, :] = 0

        cv2.putText(
            frame,
            "Agent's view",
            (6, 12), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1
        )
        cv2.putText(
            frame,
            "Omniscient view",
            (264, 12), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1
        )
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return frame

    def get_brightness(self):
        if self.time <= (self.day_len / 3): # First daytime
            return int(255 * (self.time - self.start_time) / (self.day_len / 3 - self.start_time))
        elif self.time <= (self.day_len * 2 / 3): # Night
            return 255
        else: # Second daytime
            return 255 - int(
                255 * (self.time - (self.day_len * 2 / 3)) / 
                    (self.end_time - (self.day_len * 2 / 3))
            )

    @staticmethod
    def saturate_dark(img, darker):
        pic = img.copy()
        pic = pic.astype('int32')
        pic = np.clip(pic - darker, 0, 255)
        pic = pic.astype('uint8')
        return pic


class BallCrashing(gym.Env):
    """ Ball crashing came """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        num_good: int = 3,
        num_bad: int = 3,
        reward_range: Union[Tuple, List] = (-100, 100),
        day_len: int = 300,
        render_mode: str = None
    ):
        super().__init__()
        
        assert len(reward_range) == 2, "A length of `reward_range` should be 2."
        assert reward_range[1] > 0 and reward_range[0] < 0, "`reward_range` should cover the range from negative to positive."
        print("[Ball Crashing] Ready for playing!")

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

        self.num_good = num_good
        self.num_bad = num_bad
        self.reward_range = reward_range
        self.day_len = day_len

        self.good_colors = list(COLORS.keys())[:num_good]
        self.bad_colors = list(COLORS.keys())[9:9+num_bad]

        self.ball_types = self.make_ball_types(
            self.num_good,
            self.num_bad,
            self.reward_range
        ) # {color: reward}

        self.board = Whiteboard(day_len=day_len, radius=5)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state        
        self.start_time = randrange(20)
        self.end_time = randrange(self.day_len - 20, self.day_len)
        self.cur_time = self.start_time

        self.crd = self.generate_coordinate(gen_agent=True)
        self.balls = dict() # {coordinates: color}
        for _ in range(self.num_good):
            self.make_new_ball(ball_type='good')
        for _ in range(self.num_bad):
            self.make_new_ball(ball_type='bad')

        self.board.init_board(self.start_time, self.end_time, self.crd)
        for coord, color in self.balls.items():
            self.board.add_circle(coord, COLORS[color])

        info = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "agent_pos": self.crd,
            "balls": self.balls
        }

        return self.board.get_state(), info

    def step(self, action):
        self.cur_time += 1

        if action == 0: # No-op
            pass  
        elif action == 1: # Up
            self.crd = (max(self.crd[0] - 2, 0), self.crd[1])
        elif action == 2: # Down
            self.crd = (min(self.crd[0] + 2, 83), self.crd[1])
        elif action == 3: # Left
            self.crd = (self.crd[0], max(self.crd[1] - 2, 0))
        elif action == 4: # Right
            self.crd = (self.crd[0], min(self.crd[1] + 2, 83))

        # Reward
        reward = 0
        threshold = self.board.radius // 2
        for ex_crd in list(self.balls.keys()):
            if (abs(ex_crd[0] - self.crd[0]) <= threshold) and (abs(ex_crd[1] - self.crd[1]) <= threshold):
                reward += self.ball_types[self.balls.pop(ex_crd)]
                self.board.remove_circle(ex_crd)

                new_ball = self.make_new_ball('good' if reward > 0 else 'bad')
                self.board.add_circle(new_ball[0], COLORS[new_ball[1]])
        reward -= 1  # time penalty
        self.board.move_agent(self.crd)

        # Done
        done = (self.cur_time == self.end_time)

        # Info
        info = {
            "cur_time": self.cur_time
        }

        return self.board.get_state(), reward, done, False, info

    def render(self):
        return self.board.get_frame_for_render()

    def make_new_ball(self, ball_type: str) -> Tuple[Tuple[int, int], str]:
        assert len(self.balls) <= self.num_good + self.num_bad, "Too many balls."

        # Set the coordinate
        coordinate = self.generate_coordinate(gen_agent=False)

        # Set the color
        color = None
        if ball_type == 'good':
            color = random.choice(self.good_colors)
        else:
            color = random.choice(self.bad_colors)

        self.balls.update({coordinate: color})
        return (coordinate, color)

    def generate_coordinate(
        self,
        gen_agent: bool = False
    ) -> Tuple[int, int]:
        def validate_coordinate(crd, existing_crds):
            threshold = self.board.radius * 2.5
            for ex_crd in existing_crds:
                if (abs(ex_crd[0] - crd[0]) < threshold) and (abs(ex_crd[1] - crd[1]) < threshold):
                    return False
            return True
        
        new_crd = (randrange(0, 84), randrange(0, 84))
        if not gen_agent:        
            existing_crds = list(self.balls.keys()) + [self.crd]
            while not validate_coordinate(new_crd, existing_crds):
                new_crd = (randrange(0, 84), randrange(0, 84))
        return new_crd

    def make_ball_types(
        self,
        num_good: int,
        num_bad: int,
        reward_range: Tuple[int, int]
    ) -> List[Tuple[str, float]]:
        """ Determine the color and reward of the balls to be used. """
        good_range = np.linspace(0, reward_range[-1], num_good + 1)[0:]
        bad_range = np.linspace(reward_range[0], 0, num_bad + 1)[:-1]

        candidates = dict()
        good_candidates = {name: reward for name, reward in zip(self.good_colors, good_range)}
        bad_candidates = {name: reward for name, reward in zip(self.bad_colors, bad_range)}
        
        candidates.update(good_candidates)
        candidates.update(bad_candidates)

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
        n_step=1000
    )

    display_frames_as_gif(
        frames=frames,
        fname="video.gif"
    )
