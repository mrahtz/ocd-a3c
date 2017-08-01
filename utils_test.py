#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import unittest
import matplotlib
import argparse
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from utils import copy_network, EnvWrapper
import gym
from gym.utils.play import play


class TestUtils(unittest.TestCase):

    def test(self):
        sess = tf.Session()

        inits = {}
        inits['from_scope'] = {}
        inits['to_scope'] = {}
        inits['from_scope']['w1'] = np.array([1.0, 2.0]).astype(np.float32)
        inits['from_scope']['w2'] = np.array([3.0, 4.0]).astype(np.float32)
        inits['to_scope']['w1'] = np.array([5.0, 6.0]).astype(np.float32)
        inits['to_scope']['w2'] = np.array([7.0, 8.0]).astype(np.float32)

        scopes = ['from_scope', 'to_scope']

        variables = {}
        for scope in scopes:
            with tf.variable_scope(scope):
                w1 = tf.Variable(inits[scope]['w1'], name='w1')
                w2 = tf.Variable(inits[scope]['w2'], name='w2')
                variables[scope] = {'w1': w1, 'w2': w2}

        sess.run(tf.global_variables_initializer())

        """
        Check that the variables start off being what we expect them to.
        """
        for scope in scopes:
            for var_name, var in variables[scope].items():
                actual = sess.run(var)
                if 'w1' in var_name:
                    expected = inits[scope]['w1']
                elif 'w2' in var_name:
                    expected = inits[scope]['w2']
                np.testing.assert_equal(actual, expected)

        copy_network(sess, from_scope='from_scope', to_scope='to_scope')

        """
        Check that the variables in from_scope are untouched.
        """
        for var_name, var in variables['from_scope'].items():
            actual = sess.run(var)
            if 'w1' in var_name:
                expected = inits['from_scope']['w1']
            elif 'w2' in var_name:
                expected = inits['from_scope']['w2']
            np.testing.assert_equal(actual, expected)

        """
        Check that the variables in to_scope have been modified.
        """
        for var_name, var in variables['to_scope'].items():
            actual = sess.run(var)
            if 'w1' in var_name:
                expected = inits['from_scope']['w1']
            elif 'w2' in var_name:
                expected = inits['from_scope']['w2']
            np.testing.assert_equal(actual, expected)


class DummyEnv:
    def __init__(self):
        self.i = 0
        self.observation_space = None
        self.unwrapped = None

    def reset(self):
        o = np.zeros((210, 160, 3))
        return o

    def step(self, a):
        o = np.zeros((210, 160, 3))
        # Draw a horizontal series of marks
        draw_y = 10
        draw_x = 10
        while draw_x < 160:
            o[draw_y, draw_x, 0] = 255
            draw_x += 10
        # Draw a mark below the mark corresponding
        # to the current frame
        draw_y = 20
        draw_x = 10 + self.i * 10
        o[draw_y, draw_x, 0] = 255
        self.i += 1
        return o, 0, False, None

    def render(self):
        pass


def test(env):
    o = env.reset()
    for i in range(4):
        plt.figure()
        plt.title("Frame %d" % i)
        plt.imshow(o, cmap='gray')
        o = env.step(0)[0]
    plt.show()


def test_envwrapper():
    """
    Test EnvWrapper.
    """
    print("Frame 1 mark 1, frame 2 mark 2, frame 3 mark 3")
    env = EnvWrapper(DummyEnv(), pool=False, frameskip=1)
    test(env)
    print("Frame 1 mark 1, frame 2 mark 1,2, frame 3 mark 2,3")
    env = EnvWrapper(DummyEnv(), pool=True, frameskip=1)
    test(env)
    print("Frame 1 mark 2, frame 2 mark 4, frame 3 mark 6")
    env = EnvWrapper(DummyEnv(), pool=False, frameskip=2)
    test(env)
    print("Frame 1 mark 3, frame 2 mark 6, frame 3 mark 9")
    env = EnvWrapper(DummyEnv(), pool=False, frameskip=3)
    test(env)
    print("Frame 1 mark 2+3, frame 2 mark 5+6, frame 3 mark 8+9")
    env = EnvWrapper(DummyEnv(), pool=True, frameskip=3)
    test(env)


def test_prepro():
    env = EnvWrapper(
        gym.make('Pong-v0'), pool=False, frameskip=1)
    play(env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test')
    args = parser.parse_args()
    if args.test == 'envwrapper':
        test_envwrapper()
    elif args.test == 'prepro':
        test_prepro()
    elif args.test == 'copynetwork':
        unittest.main()
