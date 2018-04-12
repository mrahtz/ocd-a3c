#!/usr/bin/env python3

import tensorflow as tf
import unittest
import numpy as np
from train_ops import create_train_ops

grad_bufs = None
sess = None


def assert_grad_bufs_zero():
    global grad_bufs
    for buf in grad_bufs.items():
        val = sess.run(buf)[0]
        np.testing.assert_equal(val, np.array([0., 0.]))


class TestTrainOps(unittest.TestCase):

    def testUnusedVariables(self):
        """
        Test whether everything behaves correctly if we have a trainable
        variable which isn't relevant to the loss function
        """
        tf.reset_default_graph()
        sess = tf.Session()

        scopes = ['update_scope', 'apply_scope']
        ops = {}
        for scope in scopes:
            ops[scope] = {}
            with tf.variable_scope(scope):
                ops[scope]['w1'] = tf.Variable(10.0)
                ops[scope]['w2'] = tf.Variable(10.0)
                ops[scope]['loss'] = ops[scope]['w1']

        o = tf.train.GradientDescentOptimizer(learning_rate=1)

        update_ops, apply_ops, zero_ops = \
            create_train_ops(ops['update_scope']['loss'],
                             o,
                             'update_scope',
                             'apply_scope')

        sess.run(tf.global_variables_initializer())
        sess.run(update_ops)
        sess.run(update_ops)
        sess.run(update_ops)
        sess.run(apply_ops)

        actual = sess.run(ops['update_scope']['w1'])
        expected = 10.0
        np.testing.assert_equal(actual, expected)

        actual = sess.run(ops['update_scope']['w2'])
        expected = 10.0
        np.testing.assert_equal(actual, expected)

        actual = sess.run(ops['apply_scope']['w1'])
        expected = 7.0
        np.testing.assert_equal(actual, expected)

        actual = sess.run(ops['apply_scope']['w2'])
        expected = 10.0
        np.testing.assert_equal(actual, expected)

    def test(self):
        global grad_bufs
        global sess

        inits = {}
        inits['w1'] = np.array([10.0, 20.0]).astype(np.float32)
        inits['w2'] = np.array([5.0, 10.0]).astype(np.float32)

        scopes = ['update_scope', 'apply_scope']

        tf.reset_default_graph()
        sess = tf.Session()

        input = tf.placeholder(tf.float32, [None, 2])

        vars = {}
        losses = {}
        for scope in scopes:
            with tf.variable_scope(scope):
                w1 = tf.Variable(inits['w1'], name='w1')
                w2 = tf.Variable(inits['w2'], name='w2')
                # NB reduce_sum is necessary to ensure that the gradients
                # accumulated for multiple examples in a batch are the same as
                # if the examples were presented in individual batches
                losses[scope] = tf.reduce_sum(w1 + input * w2, axis=-1)
                vars[scope] = {'w1': w1, 'w2': w2}

        o = tf.train.GradientDescentOptimizer(learning_rate=1)

        """
        Check that no extra trainable variables have been introduced.
        """
        # two variables, two scopes, for a total of 4 trainable variables
        assert(len(tf.trainable_variables()) == 4)

        update_ops, apply_ops, zero_ops = create_train_ops(losses['update_scope'], o, 'update_scope', 'apply_scope')

        assert(len(tf.trainable_variables()) == 4)

        sess.run(tf.global_variables_initializer())

        grad_bufs = {v.name: v for v in tf.global_variables() if 'grad_buf' in v.name}

        """
        Check that the gradient buffers start out zero.
        """
        assert_grad_bufs_zero()

        # so the first loss term looks like w1 + 1 * w2
        # and the second term looks like w1 + 2 * w2
        sess.run(update_ops, feed_dict={input: [[1, 1],
                                                [2, 2]]})

        """
        Confirm that no changes have taken place to the trainable
        variables yet in either scope.
        """
        for scope in scopes:
            for var_name, var in vars[scope].items():
                val = sess.run(var)
                np.testing.assert_equal(val, inits[var_name])

        """
        Confirm that the gradient buffers look reasonable.
        """
        for buf_name, buf in grad_bufs.items():
            actual = sess.run(buf)
            # first loss term was w1 + 1 * w2
            # second was w1 + 2 * w2
            # first loss term contribution:
            # derivative wrt to each element of both vectors should be 1
            # second loss term contribution:
            # derivative wrt w1 should be 1; derivative wrt w2 should be 2
            if 'w1' in buf_name:
                expected = np.array([1., 1.]) + np.array([1., 1.])
            elif 'w2' in buf_name:
                expected = np.array([1., 1.]) + np.array([2., 2.])
            np.testing.assert_equal(actual, expected)

        # loss will be e.g. w1 + [3, 4] * w2
        sess.run(update_ops, feed_dict={input: [[3, 4],
                                                [5, 6]]})

        """
        Confirm that the gradient buffers still look reasonable.
        """
        for buf_name, buf in grad_bufs.items():
            actual = sess.run(buf)
            if 'w1' in buf_name:
                expected = np.array([1., 1.]) + np.array([1., 1.]) + \
                           np.array([1., 1.]) + np.array([1., 1.])
            elif 'w2' in buf_name:
                expected = np.array([1., 1.]) + np.array([2., 2.]) + \
                           np.array([3., 4.]) + np.array([5., 6.])
            np.testing.assert_equal(actual, expected)

        sess.run(apply_ops)

        """
        Confirm that no changes have been made to the variables in update_scope.
        """
        for var_name, var in vars['update_scope'].items():
            actual = sess.run(var)
            if 'w1' in var_name:
                expected = inits['w1']
            elif 'w2' in var_name:
                expected = inits['w2']
            np.testing.assert_equal(actual, expected)

        """
        Confirm that changes _have_ been made to the variables in apply_scope.
        """
        for var_name, var in vars['apply_scope'].items():
            actual = sess.run(var)
            # w1 started off as [10, 20];
            # gradient wrt w1 was 1 on each step,
            # and we went for 4 steps with step size of 1
            if 'w1' in var_name:
                expected = [10 - 1. - 1. - 1. - 1.,
                            20 - 1. - 1. - 1. - 1.]
            # w2 started off as [5, 10]
            # gradients were [1, 1], [2, 2], [3, 4], and [5, 6]
            elif 'w2' in var_name:
                expected = [5.  - 1. - 2. - 3. - 5.,
                            10. - 1. - 2. - 4. - 6.]
            np.testing.assert_equal(actual, expected)

        sess.run(zero_ops)

        """
        Check that gradient buffers have been zeroed.
        """
        assert_grad_bufs_zero()


if __name__ == '__main__':
    unittest.main()
