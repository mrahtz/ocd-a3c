#!/usr/bin/env python3

import unittest

import gym
import numpy as np

from preprocessing import EnvWrapper, prepro2, NumberFrames


class DummyEnv(gym.Env):
    """
    A super-simple environment which just paints a white dot starting at (10, 10)
    and moving 10 pixels right on every step.

    If draw_n_dots is true, also indicate the current step number by the
    number of dots in each column.
    """

    OBS_DIMS = (210, 160, 3)

    def __init__(self):
        self.n_steps = None
        self.draw_n_dots = False

    def get_action_meanings(self):
        return ['NOOP']

    def reset(self):
        self.n_steps = 0
        return self._get_obs()

    def _get_obs(self):
        self.n_steps += 1
        obs = np.zeros(self.OBS_DIMS, dtype=np.uint8)
        dot_width = 3
        dot_height = 3
        # Draw the a dot on the first row
        x = 10 * self.n_steps
        y = 10
        obs[y:y + dot_height, x:x + dot_width] = 255
        if self.draw_n_dots:
            # Draw another n_steps - 1 dots in the same column,
            # for a total of n_steps dots in the column
            for i in range(1, self.n_steps):
                y = 10 + i * 10
                obs[y:y + dot_height, x:x + dot_width] = 255
        return obs

    def step(self, action):
        obs = self._get_obs()
        reward = 0
        info = None

        if self.n_steps >= 16:
            done = True
        else:
            done = False

        return obs, reward, done, info


class TestPreprocessing(unittest.TestCase):

    def check_full_preprocessing(self):
        """
        Manual check of the full set of preprocessing steps.
        Not run as part of normal unit tests; run me with
          ./preprocessing_test.py TestPreprocessing.check_full_preprocessing
        """
        from pylab import subplot, imshow, show, tight_layout
        env = DummyEnv()
        env.draw_n_dots = True
        env_wrapped = EnvWrapper(env,
                                 prepro2=prepro2,
                                 frameskip=4)

        obs1 = env_wrapped.reset()
        obs2, _, _, _ = env_wrapped.step(0)
        obs3, _, _, _ = env_wrapped.step(0)
        obs4 = env_wrapped.reset()

        subplot(4, 1, 1)
        imshow(obs1, cmap='gray')
        subplot(4, 1, 2)
        imshow(obs2, cmap='gray')
        subplot(4, 1, 3)
        imshow(obs3, cmap='gray')
        subplot(4, 1, 4)
        imshow(obs4, cmap='gray')
        tight_layout()
        show()

    def play_pong(self):
        """
        Manual check of full set of preprocessing steps for Pong.
        Not run as port of normal unit tests; run me with
          ./preprocessing_test.py TestPreprocessing.play_pong_generic_wrap
          ./preprocessing_test.py TestPreprocessing.play_pong_special_wrap
        """
        from gym.utils import play as gym_play
        env = gym.make('PongNoFrameskip-v4')
        env = NumberFrames(env)
        env_wrapped = EnvWrapper(env,
                                 prepro2=prepro2,
                                 frameskip=4)
        gym_play.play(env_wrapped, fps=15, zoom=4)


if __name__ == '__main__':
    unittest.main()
