import tensorflow as tf
import numpy as np

# each agent:
# - sample an action
# - receive a reward
# -

N_ACTIONS = 9

x = tf.placeholder(tf.float32, [None, 84, 84, 4])

x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=8,
        strides=4,
        activation=tf.nn.relu)

x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu)

x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu)

x = tf.layers.dense(
        inputs=x,
        units=512,
        activation=tf.nn.relu)

x = tf.layers.dense(
        inputs=x,
        units=N_ACTIONS,
        activation=None)


"""
def with_prob(p):
    if np.random.random() < p:
        return True
    else:
        return False

if with_prob(epsilon):
    action = random
else:
    action = max(network_outputs)

if done:
    y = reward
else:
    y = gamma * max(network_2_output)
"""
