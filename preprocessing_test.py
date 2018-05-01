#!/usr/bin/env python3

import unittest

import gym
import numpy as np

from preprocessing import EnvWrapper, prepro2, NumberFrames


class DummyEnv(gym.Env):
    """
    A super-simple environment which just paints a white dot starting at (10, 10)
    and moving 10 pixels right on every step.

    Rewards returned corresponds to the current step number. ("Just after
    reset" corresponds to step 1, so the reward from the first step taken is 2.)

    If draw_n_dots is true, also indicate the current step number by the
    number of dots in each column.
    """

    OBS_DIMS = (210, 160, 3)

    def __init__(self, dot_width=1, dot_height=1, draw_n_dots=False):
        self.step_n = None
        self.draw_n_dots = draw_n_dots
        self.dot_width = dot_width
        self.dot_height = dot_height

    def get_action_meanings(self):
        return ['NOOP']

    def reset(self):
        self.step_n = 1
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        """
        Draw a dot in a column corresponding to the current step number.
        If draw_n_dots, also draw a bunch of extra dots in the column to show
        exactly which step number we're on, such that no. of dots = step number.
        """
        obs = np.zeros(self.OBS_DIMS, dtype=np.uint8)
        w = self.dot_width
        h = self.dot_height
        # Draw the a dot on the first row
        x = 10 * self.step_n
        y = 10
        obs[y:y + h, x:x + w] = 255
        if self.draw_n_dots:
            # Draw another n_steps - 1 dots in the same column,
            # for a total of n_steps dots in the column
            for i in range(1, self.step_n):
                y = 10 + i * 10
                obs[y:y + h, x:x + w] = 255
        return obs

    def step(self, action):
        if self.step_n >= 30:
            done = True
        else:
            done = False
            self.step_n += 1

        obs = self._get_obs()
        reward = self.step_n
        info = None

        return obs, reward, done, info


class TestPreprocessing(unittest.TestCase):

    def check_full_preprocessing(self):
        """
        Manual check of the full set of preprocessing steps.
        Not run as part of normal unit tests; run me with
          ./preprocessing_test.py TestPreprocessing.check_full_preprocessing
        """
        from pylab import subplot, imshow, show, tight_layout
        env = DummyEnv(dot_width=2, dot_height=2, draw_n_dots=True)
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
