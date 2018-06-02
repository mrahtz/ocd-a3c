#!/usr/bin/env python3

import unittest

import numpy as np
import tensorflow as tf

from multi_scope_train_op import make_train_op


class TestMultiScopeTrainOp(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.Session()

    def test_gradient_clipping(self):
        with tf.variable_scope('compute_scope'):
            v_compute = tf.Variable(1.0)
            loss = tf.constant(1e6) * v_compute
        with tf.variable_scope('apply_scope'):
            v_apply = tf.Variable(1.0)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        train_op, grads_tensor = make_train_op(loss, optimizer,
                                                 'compute_scope',
                                                 'apply_scope',
                                               max_grad_norm=1e3)

        self.sess.run(tf.global_variables_initializer())
        # Without clipping, the gradient would be 1e6
        grads = self.sess.run(grads_tensor)
        self.assertAlmostEqual(grads, 1e3, places=3)

        self.sess.run(train_op)
        v_apply_val = self.sess.run(v_apply)
        # We started at 1.0, and should have taken one step of -1000
        self.assertAlmostEqual(v_apply_val, 1.0 - 1000.0, places=3)

    def test_compute_scope(self):
        """
        Test whether gradients are really calculated in the compute scope
        """
        ops = {}
        with tf.variable_scope('compute_scope'):
            w1 = tf.Variable(10.0)
            w2 = tf.Variable(5.0)
            loss = w1 * w2
            d = ops['compute_scope'] = {}
            d['w1'] = w1
            d['w2'] = w2
        with tf.variable_scope('apply_scope'):
            w1 = tf.Variable(2.0)
            w2 = tf.Variable(3.0)
            d = ops['apply_scope'] = {}
            d['w1'] = w1
            d['w2'] = w2

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        train_op, _ = make_train_op(loss, optimizer,
                                      'compute_scope', 'apply_scope')

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(train_op)

        # The gradient wrt w1 should have been -5.0,
        # and the gradient wrt w2 should have been -10.0
        w1, w2 = self.sess.run([ops['apply_scope']['w1'],
                                ops['apply_scope']['w2']])
        np.testing.assert_equal(w1, 2.0 - 5.0)
        np.testing.assert_equal(w2, 3.0 - 10.0)

    def test_unused_variables(self):
        """
        Test whether everything behaves correctly if we have a trainable
        variable which isn't relevant to the loss function
        """
        scopes = ['compute_scope', 'apply_scope']
        ops = {}
        for scope in scopes:
            ops[scope] = {}
            with tf.variable_scope(scope):
                ops[scope]['w1'] = tf.Variable(10.0)
                ops[scope]['w2'] = tf.Variable(10.0)
                ops[scope]['loss'] = ops[scope]['w1']

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)

        train_op, _ = make_train_op(ops['compute_scope']['loss'], optimizer,
                                      'compute_scope', 'apply_scope')

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(train_op)
        self.sess.run(train_op)
        self.sess.run(train_op)

        # Check that variables in compute_scope haven't been touched
        w1, w2 = self.sess.run([ops['compute_scope']['w1'],
                                ops['compute_scope']['w2']])
        np.testing.assert_equal(w1, 10.0)
        np.testing.assert_equal(w2, 10.0)

        # Check that w1 has been updated, but that w2 hasn't
        # (since it wasn't relevant to the loss)
        w1, w2 = self.sess.run([ops['apply_scope']['w1'],
                                ops['apply_scope']['w2']])
        np.testing.assert_equal(w1, 7.0)
        np.testing.assert_equal(w2, 10.0)

    def test_full_run(self):
        inits = {'w1': np.array([10.0, 20.0]).astype(np.float32),
                 'w2': np.array([5.0, 10.0]).astype(np.float32)}
        scopes = ['compute_scope', 'apply_scope']

        w2_mult = tf.placeholder(tf.float32, [None, 2])
        variables = {}
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
                variables[scope] = {'w1': w1, 'w2': w2}
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)

        # Check that no extra trainable variables are introduced
        # We have two variables, two scopes, for a total of 4 trainable
        # variables
        assert len(tf.trainable_variables()) == 4
        train_op, _ = make_train_op(losses['compute_scope'], optimizer,
                                      'compute_scope', 'apply_scope')
        assert len(tf.trainable_variables()) == 4

        self.sess.run(tf.global_variables_initializer())

        # Loss in item 0 of batch will be w1 + [1, 1] . w2
        # Loss in item 1 of batch will be w1 + [2, 2] . w2
        self.sess.run(train_op, feed_dict={w2_mult: [[1, 1],
                                                     [2, 2]]})

        # Losses will be w1 + [3, 4] . w1,
        #                w1 + [5, 6] . w2
        self.sess.run(train_op, feed_dict={w2_mult: [[3, 4],
                                                     [5, 6]]})

        # Confirm that no changes have been made to the variables in
        # compute_scope
        actual = self.sess.run(variables['compute_scope']['w1'])
        expected = inits['w1']
        np.testing.assert_equal(actual, expected)
        actual = self.sess.run(variables['compute_scope']['w2'])
        expected = inits['w2']
        np.testing.assert_equal(actual, expected)

        # Confirm that changes _have_ been made to the variables in apply_scope.
        actual = self.sess.run(variables['apply_scope']['w1'])
        # w1 started off as [10, 20];
        # gradient wrt w1 was 1 on each step,
        # and we went for 4 steps with step size of 1
        expected = [10 - 1. - 1. - 1. - 1.,
                    20 - 1. - 1. - 1. - 1.]
        np.testing.assert_equal(actual, expected)
        actual = self.sess.run(variables['apply_scope']['w2'])
        # w2 started off as [5, 10]
        # gradients were [1, 1], [2, 2], [3, 4], and [5, 6]
        expected = [5. - 1. - 2. - 3. - 5.,
                    10. - 1. - 2. - 4. - 6.]
        np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
