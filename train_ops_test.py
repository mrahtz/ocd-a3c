import tensorflow as tf
import unittest
import re
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

    def test(self):
        global grad_bufs
        global sess

        inits = {}
        inits['w1'] = np.array([10.0, 20.0]).astype(np.float32)
        inits['w2'] = np.array([5.0, 10.0]).astype(np.float32)

        scopes = ['update_scope', 'apply_scope']

        tf.reset_default_graph()
        sess = tf.Session()

        vars = {}
        losses = {}
        for scope in scopes:
            with tf.variable_scope(scope):
                w1 = tf.Variable(inits['w1'], name='w1')
                w2 = tf.Variable(inits['w2'], name='w2')
                losses[scope] = w1 + 2 * w2
                vars[scope] = {'w1': w1, 'w2': w2}

        o = tf.train.GradientDescentOptimizer(learning_rate=1)

        """
        Check that no extra trainable variables have been introduced.
        """
        # two variables, two scopes, for a total of 4 trainable variables
        assert(len(tf.trainable_variables()) == 4)

        update_ops, apply_ops, zero_ops =     create_train_ops(losses['update_scope'], o, 'update_scope', 'apply_scope')

        assert(len(tf.trainable_variables()) == 4)

        sess.run(tf.global_variables_initializer())

        grad_bufs = {v.name: v for v in tf.global_variables() if 'grad_buf' in v.name}

        """
        Check that the gradient buffers start out zero.
        """
        assert_grad_bufs_zero()

        sess.run(update_ops)

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
            # loss is w1 + 2 * w2,
            # so derivative wrt each element of w1 should be 1,
            # and derivative wrt each element of w2 should be 2
            if 'w1' in buf_name:
                expected = [1., 1.]
            elif 'w2' in buf_name:
                expected = [2., 2.]
            np.testing.assert_equal(actual, expected)

        sess.run(update_ops)

        """
        Confirm that the gradient buffers still look reasonable.
        """
        for buf_name, buf in grad_bufs.items():
            actual = sess.run(buf)
            if 'w1' in buf_name:
                expected = [2., 2.]
            elif 'w2' in buf_name:
                expected = [4., 4.]
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
            # gradient wrt w1 was 1 on each step;
            # gradient wrt w2 was 2 on each step;
            # we accumulated gradients for 2 steps;
            # we started off with [10, 20] and [5, 10];
            # and we have a step size of 1
            if 'w1' in var_name:
                expected = [8., 18.]
            elif 'w2' in var_name:
                expected = [1., 6.]
            np.testing.assert_equal(actual, expected)

        sess.run(zero_ops)

        """
        Check that gradient buffers have been zeroed.
        """
        assert_grad_bufs_zero()

if __name__ == '__main__':
    unittest.main()
