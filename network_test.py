#!/usr/bin/env python3

import unittest

import numpy as np
import tensorflow as tf

from network import Network, make_inference_network


class TestNetwork(unittest.TestCase):

    def test_policy_loss(self):
        """
        Does calculating policy loss based on the cross-entropy really give the right result?
        """
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        with tf.variable_scope('global'):
            make_inference_network(obs_shape=(84, 84, 4), n_actions=6)
        network = Network('foo_scope', n_actions=6, entropy_bonus=0.0, value_loss_coef=0.5,
                          max_grad_norm=0.5, optimizer=optimizer, add_summaries=False)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        obs = np.random.rand(3, 84, 84, 4)
        action_probs = sess.run(network.action_probs,
                                feed_dict={network.states: obs})

        rewards = [4, 5, 6]
        actions = [1, 3, 2]
        advantage, actual_loss = sess.run([network.advantage, network.policy_loss],
                                          feed_dict={network.states: obs,
                                                     network.actions: actions,
                                                     network.returns: rewards})
        expected_loss = (-np.log(action_probs[0][1]) * advantage[0] +
                         -np.log(action_probs[1][3]) * advantage[1] +
                         -np.log(action_probs[2][2]) * advantage[2])
        expected_loss /= 3
        self.assertAlmostEqual(expected_loss, actual_loss, places=5)


if __name__ == '__main__':
    unittest.main()
