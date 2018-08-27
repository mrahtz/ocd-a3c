#!/usr/bin/env python3

import os.path as osp
import subprocess
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from network import make_inference_network

"""
Reinforcement learning is really sensitive to random initialization.

Let's test that different runs really do run from the same random initialization, and that running
the current version of the code with a given seed produces the same results as for previous versions
of the code.
"""


def vars_hash_after_training(seed, n_steps):
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = ("python train.py PongNoFrameskip-v4 "
               "--manager_wake_interval 1 --seed {} --n_steps {}".format(seed, n_steps))
        cmd = cmd.split(' ') + ["--log_dir", temp_dir]
        subprocess.call(cmd)

        tf.reset_default_graph()
        sess = tf.Session()
        with tf.variable_scope('global'):
            make_inference_network(obs_shape=(84, 84, 4), n_actions=6)  # For PongNoFrameskip-v4

        saver = tf.train.Saver()
        ckpt_dir = osp.join(temp_dir, 'checkpoints')
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(sess, ckpt_file)
        tf_vars = sess.run(tf.trainable_variables())
        vars_hash = np.sum([np.sum(v) for v in tf_vars])

        return vars_hash


class TestTrain(unittest.TestCase):
    hash_10_steps = vars_hash_after_training(n_steps=10, seed=0)
    hash_100_steps_1 = vars_hash_after_training(n_steps=100, seed=0)
    hash_100_steps_2 = vars_hash_after_training(n_steps=100, seed=0)
    hash_100_steps_different_seed = vars_hash_after_training(n_steps=100, seed=1)

    def test_run_repeatability(self):
        """
        Check that if we do two runs for 100 steps starting from the same
        seed we get the same result.

        (Note that because of the asynchronous nature of A3C, this will only work if we run with a
        single worker.)
        """
        self.assertEqual(TestTrain.hash_100_steps_1, TestTrain.hash_100_steps_2)

    def test_variable_change(self):
        """
        Check that the last run didn't succeed just because somehow training didn't update variables
        at all.
        """
        self.assertNotAlmostEqual(TestTrain.hash_10_steps, TestTrain.hash_100_steps_1)

    def test_seed_change(self):
        """
        Check that changing the seed really does change the result.
        """
        self.assertNotAlmostEqual(TestTrain.hash_100_steps_1,
                                  TestTrain.hash_100_steps_different_seed)

    def test_randomness(self):
        """
        Test that randomness is set up exactly the same as it was for
        previous runs.
        """
        last_seen_var_hash = 13.793851
        self.assertAlmostEqual(TestTrain.hash_100_steps_1, last_seen_var_hash, places=5)


if __name__ == '__main__':
    unittest.main()
