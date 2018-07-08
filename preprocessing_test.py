#!/usr/bin/env python3

import unittest

import gym
import numpy as np
from numpy.testing import assert_array_equal

from debug_wrappers import NumberFrames, ConcatFrameStack
from preprocessing import MaxWrapper, FrameStackWrapper, FrameSkipWrapper, \
    ExtractLuminanceAndScaleWrapper, generic_preprocess, pong_preprocess, \
    ClipRewardsWrapper, EndEpisodeOnLifeLossWrapper

"""
Tests for preprocessing and environment tweak wrappers.
"""


class ALE:
    def __init__(self, n_lives=0):
        self.n_lives = n_lives

    def lives(self):
        return self.n_lives


class DummyEnv(gym.Env):
    """
    A super-simple environment which just paints a white dot starting at
    (10, 10) and moving 10 pixels right on every step.

    Rewards returned corresponds to the current step number. Reset counts as
    step 1, so the reward from the first step taken is 2.)

    If draw_n_dots is true, also indicate the current step number by the
    number of dots in each column.
    """

    OBS_DIMS = (210, 160, 3)

    def __init__(self, dot_width=1, dot_height=1, draw_n_dots=False):
        self.step_n = None
        self.draw_n_dots = draw_n_dots
        self.dot_width = dot_width
        self.dot_height = dot_height
        self.ale = ALE()

    @staticmethod
    def get_action_meanings():
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


class ConstantRewardEnv(gym.Env):
    """
    Return a specified reward on every step.
    """

    def __init__(self, reward_val=0):
        self.reward_val = reward_val

    def reset(self):
        return None

    def step(self, action):
        obs = None
        reward = self.reward_val
        info = None
        done = False

        return obs, reward, info, done


class LivesEnv(gym.Env):
    """
    Emulates env.ale.lives(). Starts with 2 lives. Loses a life every 5 steps.
    """

    def __init__(self):
        self.ale = None
        self.step_n = None

    def reset(self):
        self.ale = ALE(n_lives=2)
        self.step_n = 0
        obs = self.step_n
        self.step_n += 1
        return obs

    def step(self, action):
        if self.step_n != 0 and self.step_n % 5 == 0:
            self.ale.n_lives -= 1
        if self.ale.n_lives == 0:
            done = True
        else:
            done = False

        obs = self.step_n
        reward = None
        info = None

        self.step_n += 1

        return obs, reward, done, info


class TestPreprocessing(unittest.TestCase):

    def test_max_wrapper(self):
        env = DummyEnv()
        env_wrapped = MaxWrapper(env)

        actual_obs = env_wrapped.reset()
        # We expect to see a frame which is the maximum of frames 0 and 1
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 10] = 255
        expected_obs[10, 20] = 255
        assert_array_equal(actual_obs, expected_obs)

        # Then frames 1 and 2
        actual_obs, _, _, _ = env_wrapped.step(0)
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 20] = 255
        expected_obs[10, 30] = 255
        assert_array_equal(actual_obs, expected_obs)

        # Then frames 2 and 3
        actual_obs, _, _, _ = env_wrapped.step(0)
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 30] = 255
        expected_obs[10, 40] = 255
        assert_array_equal(actual_obs, expected_obs)

        # If we reset, we should see frames 0 and 1 again
        actual_obs = env_wrapped.reset()
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 10] = 255
        expected_obs[10, 20] = 255
        assert_array_equal(actual_obs, expected_obs)

    def test_extract_luminance_and_scale_wrapper(self):
        env = DummyEnv()
        env_wrapped = ExtractLuminanceAndScaleWrapper(env)

        # We should only have one colour channel now (luminance), with a size
        # of 84 x 84
        obs = env_wrapped.reset()
        self.assertEqual(obs.shape, (84, 84))

        obs, _, _, _ = env_wrapped.step(0)
        self.assertEqual(obs.shape, (84, 84))

    def test_frame_stack_wrapper(self):
        env = DummyEnv()
        env_wrapped = FrameStackWrapper(env)

        actual_obs = env_wrapped.reset()
        # We expect to see a stack of frames 0 to 3
        expected_obs = np.zeros((4,) + DummyEnv.OBS_DIMS, dtype=np.uint8)
        for frame_n, x in enumerate([10, 20, 30, 40]):
            expected_obs[frame_n, 10, x] = 255
        assert_array_equal(actual_obs, expected_obs)

        # Then frames 1 to 4
        actual_obs, _, _, _ = env_wrapped.step(0)
        expected_obs = np.zeros((4,) + DummyEnv.OBS_DIMS, dtype=np.uint8)
        for frame_n, x in enumerate([20, 30, 40, 50]):
            expected_obs[frame_n, 10, x] = 255
        assert_array_equal(actual_obs, expected_obs)

        # Then frames 2 to 5
        actual_obs, _, _, _ = env_wrapped.step(0)
        expected_obs = np.zeros((4,) + DummyEnv.OBS_DIMS, dtype=np.uint8)
        for frame_n, x in enumerate([30, 40, 50, 60]):
            expected_obs[frame_n, 10, x] = 255
        assert_array_equal(actual_obs, expected_obs)

        # If we reset, we should see frames 0 to 3 again
        actual_obs = env_wrapped.reset()
        expected_obs = np.zeros((4,) + DummyEnv.OBS_DIMS, dtype=np.uint8)
        for frame_n, x in enumerate([10, 20, 30, 40]):
            expected_obs[frame_n, 10, x] = 255
        assert_array_equal(actual_obs, expected_obs)

    def test_frame_skip_wrapper(self):
        env = DummyEnv()
        env_wrapped = FrameSkipWrapper(env)

        actual_obs = env_wrapped.reset()
        # We expect to see frame 0
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 10] = 255
        assert_array_equal(actual_obs, expected_obs)

        # Then frame 4
        actual_obs, _, _, _ = env_wrapped.step(0)
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 50] = 255
        assert_array_equal(actual_obs, expected_obs)

        # Then frame 8
        actual_obs, _, _, _ = env_wrapped.step(0)
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 90] = 255
        assert_array_equal(actual_obs, expected_obs)

        # If we reset, we should see frame 0 again
        actual_obs = env_wrapped.reset()
        expected_obs = np.zeros(DummyEnv.OBS_DIMS, dtype=np.uint8)
        expected_obs[10, 10] = 255
        assert_array_equal(actual_obs, expected_obs)

    def test_rewards_wrapper(self):
        env = ConstantRewardEnv()
        env_wrapped = ClipRewardsWrapper(env)
        env_wrapped.reset()

        env_wrapped.reward_val = 0
        _, reward, _, _ = env_wrapped.step(action=0)
        self.assertEqual(reward, 0)

        env.reward_val = 1
        _, reward, _, _ = env_wrapped.step(action=0)
        self.assertEqual(reward, 1)

        env.reward_val = 2
        _, reward, _, _ = env_wrapped.step(action=0)
        self.assertEqual(reward, 1)

        env.reward_val = -1
        _, reward, _, _ = env_wrapped.step(action=0)
        self.assertEqual(reward, -1)

        env.reward_val = -2
        _, reward, _, _ = env_wrapped.step(action=0)
        self.assertEqual(reward, -1)

    def test_life_loss_episode_end_wrapper(self):
        env = LivesEnv()
        env_wrapped = EndEpisodeOnLifeLossWrapper(env)

        env_wrapped.reset()

        # Two lives left

        for n in range(1, 5):
            obs, _, done, _ = env_wrapped.step(0)
            self.assertEqual(obs, n)
            self.assertFalse(done)

        obs, _, done, _ = env_wrapped.step(0)
        self.assertEqual(obs, 5)
        self.assertTrue(done)

        obs = env_wrapped.reset()
        # We should have just received the 'done' signal because of life loss,
        # so we should continue getting observations from the same actual
        # episode
        self.assertEqual(obs, 5)

        # One life left

        for n in range(4):
            obs, _, done, _ = env_wrapped.step(0)
            self.assertEqual(obs, 6 + n)
            self.assertFalse(done)

        obs, _, done, _ = env_wrapped.step(0)
        self.assertTrue(done)
        self.assertEqual(obs, 10)

        obs = env_wrapped.reset()
        # This time, we should have got the 'done' signal because we really
        # finished the episode - so we should get an observation from a new
        # episode this time.
        self.assertEqual(obs, 0)

    def test_full_preprocessing_rewards(self):
        env = DummyEnv()
        env_wrapped = generic_preprocess(env, max_n_noops=0, clip_rewards=False)
        env_wrapped.reset()
        _, r1, _, _ = env_wrapped.step(0)
        _, r2, _, _ = env_wrapped.step(0)
        _, r3, _, _ = env_wrapped.step(0)
        # MaxWrapper skips the first step after reset (which gives reward 2)
        # FrameStackWrapper does another 3 steps after reset, each of which
        # does 4 steps in the raw environment because of FrameSkipWrapper.
        # Step 1: 3, 4, 5, 6
        # Step 2: 7, 8, 9, 10
        # Step 3: 11, 12, 13, 14
        # The first step we do should get rewards 15, 16, 17 18, summed by
        # FrameSkipWrapper.
        self.assertEqual(r1, 66)
        # Then 19 + 20 + 21 + 22.
        self.assertEqual(r2, 82)
        # Then 23 + 24 + 25 + 27.
        self.assertEqual(r3, 98)

    @staticmethod
    def check_full_preprocessing():
        """
        Manual check of the full set of preprocessing steps.
        Not run as part of normal unit tests; run me with
          ./preprocessing_test.py TestPreprocessing.check_full_preprocessing
        """
        from pylab import subplot, imshow, show, tight_layout
        env = DummyEnv(dot_width=2, dot_height=2, draw_n_dots=True)
        env = NumberFrames(env)
        env_wrapped = generic_preprocess(env, max_n_noops=0)

        obs1 = env_wrapped.reset()
        obs2, _, _, _ = env_wrapped.step(0)
        obs3, _, _, _ = env_wrapped.step(0)
        obs4 = env_wrapped.reset()

        subplot(4, 1, 1)
        imshow(np.hstack(obs1), cmap='gray')
        subplot(4, 1, 2)
        imshow(np.hstack(obs2), cmap='gray')
        subplot(4, 1, 3)
        imshow(np.hstack(obs3), cmap='gray')
        subplot(4, 1, 4)
        imshow(np.hstack(obs4), cmap='gray')
        tight_layout()
        show()

    def play_pong_generic_wrap(self):
        self.play_pong(generic_preprocess)

    def play_pong_special_wrap(self):
        self.play_pong(pong_preprocess)

    @staticmethod
    def play_pong(wrap_fn):
        """
        Manual check of full set of preprocessing steps for Pong.
        Not run as poat of normal unit tests; run me with
          ./preprocessing_test.py TestPreprocessing.play_pong_generic_wrap
          ./preprocessing_test.py TestPreprocessing.play_pong_special_wrap
        """
        from gym.utils import play as gym_play
        env = gym.make('PongNoFrameskip-v4')
        env = NumberFrames(env)
        env = wrap_fn(env, max_n_noops=0)
        env = ConcatFrameStack(env)
        gym_play.play(env, fps=15, zoom=4)


if __name__ == '__main__':
    unittest.main()
