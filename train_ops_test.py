#!/usr/bin/env python3

import tensorflow as tf
import unittest
import numpy as np
from train_ops import create_train_ops


class TestTrainOps(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.Session()

    def test_unused_variables(self):
        """
        Test whether everything behaves correctly if we have a trainable
        variable which isn't relevant to the loss function
        """
        scopes = ['update_scope', 'apply_scope']
        ops = {}
        for scope in scopes:
            ops[scope] = {}
            with tf.variable_scope(scope):
                ops[scope]['w1'] = tf.Variable(10.0)
                ops[scope]['w2'] = tf.Variable(10.0)
                ops[scope]['loss'] = ops[scope]['w1']

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)

        update_ops, apply_ops, zero_ops, _ = create_train_ops(
            loss=ops['update_scope']['loss'],
            optimizer=optimizer,
            max_grad_norm=None,
            update_scope='update_scope',
            apply_scope='apply_scope')

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(update_ops)
        self.sess.run(update_ops)
        self.sess.run(update_ops)
        self.sess.run(apply_ops)

        # Check that variables in update_scope haven't been touched
        w1, w2 = self.sess.run([ops['update_scope']['w1'],
                                ops['update_scope']['w2']])
        np.testing.assert_equal(w1, 10.0)
        np.testing.assert_equal(w2, 10.0)

        # Check that w1 has been updated, but that w2 hasn't
        # (since it wasn't relevant to the loss)
        w1, w2 = self.sess.run([ops['apply_scope']['w1'],
                                ops['apply_scope']['w2']])
        np.testing.assert_equal(w1, 7.0)
        np.testing.assert_equal(w2, 10.0)

    def assert_grad_bufs_zero(self, grad_bufs):
        for buf in grad_bufs.values():
            val = self.sess.run(buf)[0]
            np.testing.assert_equal(val, np.array([0., 0.]))

    def test_full_run(self):
        global grad_bufs
        global sess

        inits = {}
        inits['w1'] = np.array([10.0, 20.0]).astype(np.float32)
        inits['w2'] = np.array([5.0, 10.0]).astype(np.float32)

        scopes = ['update_scope', 'apply_scope']

        w2_mult = tf.placeholder(tf.float32, [None, 2])
        vars = {}
        losses = {}
        # Create two variables in two scopes
        # Each variable is a vector with two elements
        for scope in scopes:
            with tf.variable_scope(scope):
                w1 = tf.Variable(inits['w1'], name='w1')
                w2 = tf.Variable(inits['w2'], name='w2')
                # NB reduce_sum is necessary to ensure that the gradients
                # accumulated for multiple examples in a batch are the same as
                # if the examples were presented in individual batches
                losses[scope] = tf.reduce_sum(w1 + w2_mult * w2, axis=-1)
                vars[scope] = {'w1': w1, 'w2': w2}
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)

        # Check that no extra trainable variables are introduced
        # We have two variables, two scopes, for a total of 4 trainable
        # variables
        assert len(tf.trainable_variables()) == 4
        update_ops, apply_ops, zero_ops, grad_bufs = create_train_ops(
            loss=losses['update_scope'],
            optimizer=optimizer,
            max_grad_norm=None,
            update_scope='update_scope',
            apply_scope='apply_scope')
        assert len(tf.trainable_variables()) == 4

        self.sess.run(tf.global_variables_initializer())

        # Check that the gradient buffers start out zero
        self.assert_grad_bufs_zero(grad_bufs)

        # Add one set of gradients to the gradient buffers

        # Loss in element 0 of batch will be w1 + [1, 1] . w2
        # Loss in element 1 of batch will be w1 + [2, 2] . w2
        self.sess.run(update_ops, feed_dict={w2_mult: [[1, 1],
                                                       [2, 2]]})

        # Confirm that no changes have taken place to the trainable
        # variables yet in either scope
        for scope in scopes:
            for var_name, var in vars[scope].items():
                val = self.sess.run(var)
                np.testing.assert_equal(val, inits[var_name])

        # Confirm that the gradient buffers have the right values.
        # Loss term in element 0 of the batch was w1 + [1, 1] . w2.
        # Derivative of loss wrt to each element of both w1 and w2 should be 1.
        # In element 1, was w1 + [2, 2] . w2.
        # Derivative should be 1 for each element of w1,
        # and 2 for each element of w2.
        # Total derivatives accumulated for each element of w1: 1 + 1 = 2.
        actual = self.sess.run(grad_bufs['w1'])
        expected = np.array([1. + 1.,
                             1. + 1.])
        np.testing.assert_equal(actual, expected)
        # Total derivatives accumulated for each element of w2: 1 + 2 = 3.
        actual = self.sess.run(grad_bufs['w2'])
        expected = np.array([1. + 2.,
                             1. + 2.])
        np.testing.assert_equal(actual, expected)

        # Add another set of gradients to the gradient buffers

        # Losses will be w1 + [3, 4] . w1,
        #                w1 + [5, 6] . w2
        self.sess.run(update_ops, feed_dict={w2_mult: [[3, 4],
                                                       [5, 6]]})

        # Confirm that the gradient buffers are still right.
        actual = self.sess.run(grad_bufs['w1'])
        expected = np.array([1. + 1. + 1. + 1.,
                             1. + 1. + 1. + 1.])
        np.testing.assert_equal(actual, expected)
        actual = self.sess.run(grad_bufs['w2'])
        expected = np.array([1. + 2. + 3. + 5.,
                             1. + 2. + 4. + 6.])
        np.testing.assert_equal(actual, expected)

        # Apply the gradient buffers

        self.sess.run(apply_ops)

        # Confirm that no changes have been made to the variables in
        # update_scope
        actual = self.sess.run(vars['update_scope']['w1'])
        expected = inits['w1']
        np.testing.assert_equal(actual, expected)
        actual = self.sess.run(vars['update_scope']['w2'])
        expected = inits['w2']
        np.testing.assert_equal(actual, expected)

        # Confirm that changes _have_ been made to the variables in apply_scope.
        actual = self.sess.run(vars['apply_scope']['w1'])
        # w1 started off as [10, 20];
        # gradient wrt w1 was 1 on each step,
        # and we went for 4 steps with step size of 1
        expected = [10 - 1. - 1. - 1. - 1.,
                    20 - 1. - 1. - 1. - 1.]
        np.testing.assert_equal(actual, expected)
        actual = self.sess.run(vars['apply_scope']['w2'])
        # w2 started off as [5, 10]
        # gradients were [1, 1], [2, 2], [3, 4], and [5, 6]
        expected = [5. - 1. - 2. - 3. - 5.,
                    10. - 1. - 2. - 4. - 6.]
        np.testing.assert_equal(actual, expected)

        # Check that zeroing the gradient buffers works.
        self.sess.run(zero_ops)
        self.assert_grad_bufs_zero(grad_bufs)


if __name__ == '__main__':
    unittest.main()
