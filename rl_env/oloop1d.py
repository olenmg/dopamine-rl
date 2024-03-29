import pickle
from random import randrange

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import imageio


class OpenLoop1DTrack(gym.Env):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, water_spout, video_path, visual_noise=False, render_mode=None):
        super().__init__()
        print("[Mouse VR] Ready for the OpenLoop 1D Track")

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(160, 210, 3), dtype=np.uint8
        )

        self.water_spout = water_spout
        self.video_path = video_path
        self.visual_noise = visual_noise #TODO

        self.screen = self._load_screen()
        self.cur_time = randrange(50) # Remove time-bias
        self.start_time = self.cur_time
        self.end_time = self.screen.shape[0] - randrange(1, 10) # Black screen
        self.state = self.screen[self.cur_time]
        self.licking_cnt = 0

        # For rendering
        self.frames = []
        self.original_frames = self._get_original_video_frames() # (682, 1288)
        self.mice_pic = self._load_mice_image()
        self.render_mode = render_mode

        # For plotting
        self.lick_timing = []
        self.lick_timing_eps = []

    def step(self, action):
        # Execute one time step within the environment
        self.cur_time += 1

        # Next state
        next_state = self.screen[self.cur_time]
        # if self.visual_noise: #TODO
        self.state = next_state

        # Reward
        reward = 0
        if action == 1:
            self._licking()
            if (self.cur_time >= self.water_spout) and (self.licking_cnt <= 20):
                reward = 10
            else:
                reward = -5

        # Done
        done = (self.cur_time == self.end_time)

        # Info
        info = {
            "cur_time": self.cur_time,
            "licking_cnt": self.licking_cnt,
            "lick_timing_eps": self.lick_timing_eps
        }

        # Water Spout rendering
        if self.cur_time == self.water_spout:
            self.bar_states.append((140., 1., True))

        return next_state, reward, done, False, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.cur_time = randrange(50)
        self.start_time = self.cur_time
        self.end_time = self.screen.shape[0] - randrange(1, 10)
        self.state = self.screen[self.cur_time]
        self.licking_cnt = 0

        self.bar_states = []

        self.lick_timing.append(self.lick_timing_eps)
        self.lick_timing_eps = []

        info = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "water_spout": self.water_spout
        }
        return self.state, info

    def render(self):
        # Render the environment to the screen
        rgb_array = self.original_frames[self.cur_time].copy()
        height, width, _ = rgb_array.shape

        pos_template = (width - 40) / (self.end_time - self.start_time)

        # Upper padding
        padding_height = height // 8
        rgb_array = cv2.copyMakeBorder(rgb_array, padding_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Base line
        base_height = padding_height // 2
        cv2.line(rgb_array, (20, base_height), (width - 20, base_height), (0, 0, 0), 2)
        
        # Current position
        x_offset = int((self.cur_time - self.start_time) * pos_template)
        y_offset = base_height - 20
        rgb_array[y_offset:y_offset+self.mice_pic.shape[0], x_offset:x_offset+self.mice_pic.shape[1], :] = self.mice_pic

        # Licking
        for lick_timing in self.lick_timing_eps:
            lick_x_pos = 20 + int((lick_timing - self.start_time) * pos_template)
            cv2.line(rgb_array, (lick_x_pos, base_height - 30), (lick_x_pos, base_height + 30), (0, 0, 0), 1)

        # Water spout
        spout_x_pos = 20 + int((self.water_spout - self.start_time) * pos_template)
        cv2.line(rgb_array, (spout_x_pos, base_height - 30), (spout_x_pos, base_height + 30), (255, 0, 0), 3)

        if self.render_mode == 'human':
            cv2.imshow("licking", rgb_array)
            cv2.waitKey(1)
        elif self.render_mode == 'gif':
            self.frames.append(cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB))
        elif self.render_mode == 'mp4':
            self.frames.append(rgb_array)
        elif self.render_mode == 'rgb_array':
            return cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)

    def cvt_screen_gray_scale(self):
        self.screen = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in self.screen])

    def save_gif(self):
        imageio.mimsave('video.gif', self.frames, duration=0.005)

    def save_mp4(self, name="test.mp4"):
        height, width, _ = self.frames[0].shape
        fps = 60

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(name, fourcc, float(fps), (width, height))
        for frame in self.frames:
            video.write(frame)
        video.release()

    def _licking(self):
        self.licking_cnt += 1
        self.lick_timing_eps.append(self.cur_time)

    def _get_original_video_frames(self):
        print("Loading VR frames...")
        capture = cv2.VideoCapture(self.video_path)

        frames = []
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        frames = np.stack(frames, axis=0)

        return frames

    @staticmethod
    def _load_mice_image():
        mice_pic = cv2.imread("assets/mice.png", cv2.IMREAD_UNCHANGED)
        mice_pic = cv2.resize(mice_pic, (40, 40))
        mice_pic = np.repeat((mice_pic[:, :, 3] < 50).reshape(40, 40, 1), repeats=3, axis=2) * 255
        return mice_pic

    @staticmethod
    def _load_screen():
        raise NotImplementedError


class OpenLoopStandard1DTrack(OpenLoop1DTrack):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, visual_noise=False, render_mode=None):
        super().__init__(
            water_spout=335,
            video_path="rl_env/track/VR_standard.mp4",
            visual_noise=visual_noise,
            render_mode=render_mode
        )

    @staticmethod
    def _load_screen():
        with open(f"rl_env/track/oloop_standard_1d.pkl", "rb") as f:
            screen = pickle.load(f)
        return screen


class OpenLoopTeleportLong1DTrack(OpenLoop1DTrack):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, visual_noise=False, render_mode=None):
        super().__init__(
            water_spout=227,
            video_path="rl_env/track/VR_teleport.mp4",
            visual_noise=visual_noise,
            render_mode=render_mode
        )

    @staticmethod
    def _load_screen():
        with open("rl_env/track/oloop_teleport_long_1d.pkl", "rb") as f:
            screen = pickle.load(f)
        return screen