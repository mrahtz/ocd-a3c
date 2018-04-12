#!/usr/bin/env python3

import unittest

import numpy as np
import tensorflow as tf

from utils import copy_network, entropy


class TestEntropy(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def test_basic(self):
        logits = [1., 2., 3., 4.]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        expected_entropy = -np.sum(probs * np.log(probs))
        actual_entropy = self.sess.run(entropy(logits))
        np.testing.assert_approx_equal(actual_entropy, expected_entropy,
                                       significant=4)

    def test_batch(self):
        # shape is 2 (batch size) x 4
        logits = [[1., 2., 3., 4.],
                  [1., 2., 2., 1.]]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        expected_entropy = -np.sum(probs * np.log(probs), axis=1, keepdims=True)
        actual_entropy = self.sess.run(entropy(logits))
        np.testing.assert_allclose(actual_entropy, expected_entropy, atol=1e-4)

    def test_gradient_descent(self):
        logits = tf.Variable([1., 2., 3., 4., 5.])
        neg_ent = -entropy(logits)
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


if __name__ == '__main__':
    unittest.main()
