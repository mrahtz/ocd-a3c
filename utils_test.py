import tensorflow as tf
import numpy as np
import unittest
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from utils import copy_network
from utils import get_o


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
    
    def step(self, a):
        o = np.zeros((210, 160, 3))
        draw_y = 10
        draw_x = 10 + self.i * 20
        o[draw_y, draw_x] = 255
        self.i += 1
        return o, 0, False, None

    def render(self):
        pass

def test_get_o():
    """
    Test get_o().
    
    Frame 0: should be just one dot, top right.
    Frames 1-3: should be two dots
                (because max is taken with previous frame),
                moving to the right
    """
    env = DummyEnv()
    get_o.last_frame = None
    o, r, done = get_o(env, 0)
    for i in range(4):
        plt.figure()
        plt.title("Frame %d" % i)
        plt.imshow(o[:, :, i])
    plt.show()

if __name__ == '__main__':
    test_get_o()
    unittest.main()
