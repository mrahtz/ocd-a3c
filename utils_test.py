#!/usr/bin/env python3

import argparse
import random
import socket
import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import create_copy_ops, logit_entropy, rewards_to_discounted_returns, \
    get_port_range, EnvWrapper, set_random_seeds


class TestMiscUtils(unittest.TestCase):

    def test_returns_easy(self):
        r = [0, 0, 0, 5]
        discounted_r = rewards_to_discounted_returns(r, discount_factor=0.99)
        np.testing.assert_allclose(discounted_r,
                                   [0.99 ** 3 * 5,
                                    0.99 ** 2 * 5,
                                    0.99 ** 1 * 5,
                                    0.99 ** 0 * 5])

    def test_returns_hard(self):
        r = [1, 2, 3, 4]
        discounted_r = rewards_to_discounted_returns(r, discount_factor=0.99)
        expected = [1 + 0.99 * 2 + 0.99 ** 2 * 3 + 0.99 ** 3 * 4,
                    2 + 0.99 * 3 + 0.99 ** 2 * 4,
                    3 + 0.99 * 4,
                    4]
        np.testing.assert_allclose(discounted_r, expected)

    def test_get_port_range(self):
        # Test 1: if we ask for 3 ports starting from port 60000
        # (which nothing should be listening on), we should get back
        # 60000, 60001 and 60002
        ports = get_port_range(60000, 3)
        self.assertEqual(ports, [60000, 60001, 60002])

        # Test 2: if we set something listening on port 60000
        # then ask for the same ports as in test 1,
        # the function should skip over 60000 and give us the next
        # three ports
        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.bind(("127.0.0.1", 60000))
        ports = get_port_range(60000, 3)
        self.assertEqual(ports, [60001, 60002, 60003])

        # Test 3: if we set something listening on port 60002,
        # the function should realise it can't allocate a continuous
        # range starting from 60000 and should give us a range starting
        # from 60003
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.bind(("127.0.0.1", 60002))
        ports = get_port_range(60000, 3)
        self.assertEqual(ports, [60003, 60004, 60005])

        s2.close()
        s1.close()

    def test_random_seed(self):
        # Note: TensorFlow random seeding doesn't work completely as expected.
        # tf.set_random_seed sets a the graph-level seed in the current graph.
        # But operations also have their own operation-level seed, which is
        # chosen deterministically based on the graph-level seed, but also
        # based on other things.
        #
        # So if you create multiple operations in the same graph,
        # each one will be given a different operation-level seed.
        # The  graph-level seed just determines what the sequence of
        # operation-level seeds will be.
        #
        # To get a bunch of operations with the same sequence of
        # operation-level seeds, we need to reset the graph before creation
        # of each bunch of operations.

        # Generate some random numbers from a specific seed
        tf.reset_default_graph()
        sess = tf.Session()
        set_random_seeds(0)
        tf_rand_var = tf.random_normal([10])
        numpy_rand_1 = np.random.rand(10)
        numpy_rand_2 = np.random.rand(10)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 numpy_rand_1, numpy_rand_2)
        tensorflow_rand_1 = sess.run(tf_rand_var)
        tensorflow_rand_2 = sess.run(tf_rand_var)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tensorflow_rand_1, tensorflow_rand_2)
        python_rand_1 = [random.random() for _ in range(10)]
        python_rand_2 = [random.random() for _ in range(10)]
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 python_rand_1, python_rand_2)

        # Put the seed back and check we get the same numbers
        tf.reset_default_graph()
        sess = tf.Session()
        set_random_seeds(0)
        tf_rand_var = tf.random_normal([10])
        numpy_rand_3 = np.random.rand(10)
        numpy_rand_4 = np.random.rand(10)
        np.testing.assert_equal(numpy_rand_1, numpy_rand_3)
        np.testing.assert_equal(numpy_rand_2, numpy_rand_4)
        tensorflow_rand_3 = sess.run(tf_rand_var)
        tensorflow_rand_4 = sess.run(tf_rand_var)
        np.testing.assert_equal(tensorflow_rand_1, tensorflow_rand_3)
        np.testing.assert_equal(tensorflow_rand_2, tensorflow_rand_4)
        python_rand_3 = [random.random() for _ in range(10)]
        python_rand_4 = [random.random() for _ in range(10)]
        np.testing.assert_equal(python_rand_1, python_rand_3)
        np.testing.assert_equal(python_rand_2, python_rand_4)

        # Set a different seed and make sure we get different numbers
        set_random_seeds(1)
        numpy_rand_5 = np.random.rand(10)
        numpy_rand_6 = np.random.rand(10)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 numpy_rand_5, numpy_rand_1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 numpy_rand_6, numpy_rand_2)
        tensorflow_rand_5 = sess.run(tf_rand_var)
        tensorflow_rand_6 = sess.run(tf_rand_var)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tensorflow_rand_5, tensorflow_rand_1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tensorflow_rand_6, tensorflow_rand_2)
        python_rand_5 = [random.random() for _ in range(10)]
        python_rand_6 = [random.random() for _ in range(10)]
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 python_rand_5, python_rand_1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 python_rand_6, python_rand_2)


class TestEntropy(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def test_basic(self):
        """
        Manually calculate entropy, and check the result matches.
        """
        logits = np.array([1., 2., 3., 4.])
        probs = np.exp(logits) / np.sum(np.exp(logits))
        expected_entropy = -np.sum(probs * np.log(probs))
        actual_entropy = self.sess.run(logit_entropy(logits))
        np.testing.assert_approx_equal(actual_entropy, expected_entropy,
                                       significant=5)

    def test_stability(self):
        """
        Test an example which would normally break numerical stability.
        """
        logits = np.array([0., 1000.])
        expected_entropy = 0.
        actual_entropy = self.sess.run(logit_entropy(logits))
        np.testing.assert_approx_equal(actual_entropy, expected_entropy,
                                       significant=5)

    def test_batch(self):
        """
        Make sure we get the right result if calculating entropies on a batch
        of probabilities.
        """
        # shape is 2 (batch size) x 4
        logits = np.array([[1., 2., 3., 4.],
                           [1., 2., 2., 1.]])
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        expected_entropy = -np.sum(probs * np.log(probs), axis=1, keepdims=True)
        actual_entropy = self.sess.run(logit_entropy(logits))
        np.testing.assert_allclose(actual_entropy, expected_entropy,
                                   atol=1e-4)

    def test_gradient_descent(self):
        """
        Check that if we start with a distribution and use gradient descent
        to maximise entropy, we end up with a maximise entropy distribution.
        """
        logits = tf.Variable([1., 2., 3., 4., 5.])
        neg_ent = -logit_entropy(logits)
        train_op = tf.train.AdamOptimizer().minimize(neg_ent)
        self.sess.run(tf.global_variables_initializer())
        for i in range(10000):
            self.sess.run(train_op)
        expected = [0.2, 0.2, 0.2, 0.2, 0.2]  # maximum entropy distribution
        actual = self.sess.run(tf.nn.softmax(logits))
        np.testing.assert_allclose(actual, expected, atol=1e-4)


class TestCopyNetwork(unittest.TestCase):

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
        copy_ops = create_copy_ops(from_scope='from_scope', to_scope='to_scope')

        sess.run(tf.global_variables_initializer())

        # Check that the variables start off being what we expect them to.
        for scope in scopes:
            for var_name, var in variables[scope].items():
                actual = sess.run(var)
                if 'w1' in var_name:
                    expected = inits[scope]['w1']
                elif 'w2' in var_name:
                    expected = inits[scope]['w2']
                np.testing.assert_equal(actual, expected)

        sess.run(copy_ops)

        # Check that the variables in from_scope are untouched.
        for var_name, var in variables['from_scope'].items():
            actual = sess.run(var)
            if 'w1' in var_name:
                expected = inits['from_scope']['w1']
            elif 'w2' in var_name:
                expected = inits['from_scope']['w2']
            np.testing.assert_equal(actual, expected)

        # Check that the variables in to_scope have been modified.
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
    elif args.test == 'copynetwork' or args.test == 'entropy':
        unittest.main(argv=[''])
