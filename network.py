from collections import namedtuple

import tensorflow as tf

from utils import logit_entropy

BETA = 0.01

Network = namedtuple('Network',
                     's a r a_softmax graph_v policy_loss value_loss loss '
                     'policy_entropy advantage')


def create_network(scope, n_actions, debug=False):
    with tf.variable_scope(scope):
        graph_s = tf.placeholder(tf.float32, [None, 84, 84, 4])
        graph_action = tf.placeholder(tf.int64, [None])
        graph_r = tf.placeholder(tf.float32, [None])

        x = tf.layers.conv2d(
            inputs=graph_s,
            filters=32,
            kernel_size=8,
            strides=4,
            activation=tf.nn.relu)

        if debug:
            # Dump observations as fed into the network to stderr,
            # for viewing with show_observations.py.
            x = tf.Print(x, [graph_s],
                         message='\ndebug observations:',
                         # max no. of values to display; max int32
                         summarize=2147483647)

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

        w, h, f = x.get_shape()[1:]
        x = tf.reshape(x, [-1, int(w * h * f)])

        x = tf.layers.dense(
            inputs=x,
            units=512,
            activation=tf.nn.relu)

        a_logits = tf.layers.dense(
            inputs=x,
            units=n_actions,
            activation=None)

        a_softmax = tf.nn.softmax(a_logits)

        graph_v = tf.layers.dense(
            inputs=x,
            units=1,
            activation=None)
        # Shape is currently (?, 1)
        # Convert to just (?)
        graph_v = graph_v[:, 0]

        advantage = graph_r - graph_v

        if debug:
            advantage = tf.Print(advantage, [graph_r],
                                 message='\ndebug returns:',
                                 summarize=2147483647)

        p = 0
        for i in range(n_actions):
            p += tf.cast(tf.equal(graph_action, i), tf.float32) * a_softmax[:,
                                                                  i]

        if debug:
            p = tf.Print(p, [graph_action],
                         message='\ndebug actions:',
                         summarize=2147483647)

        # Log probability: higher is better for actions we want to encourage
        # Negative log probability: lower is better for actions we want to
        #                           encourage
        # 1e-7: prevent log(0)
        nlp = -tf.log(p + 1e-7)

        check_nlp = tf.assert_rank(nlp, 1)
        check_advantage = tf.assert_rank(advantage, 1)
        with tf.control_dependencies([check_nlp, check_advantage]):
            # Note that the advantage is treated as a constant for the
            # policy network update step.
            # Note also that we're calculating advantages on-the-fly using
            # the value approximator. This might make us worry: what if we're
            # using the loss for training, and the advantages are calculated
            # /after/ training has changed the network? But for A3C, we don't
            # need to worry, because we compute the gradients seperately from
            # applying them.
            policy_loss = nlp * tf.stop_gradient(advantage)
            policy_loss = tf.reduce_mean(policy_loss)

            policy_entropy = tf.reduce_mean(logit_entropy(a_logits))
            # We want to maximise entropy, which is the same as
            # minimising negative entropy
            policy_loss -= BETA * policy_entropy

            value_loss = advantage ** 2
            value_loss = tf.reduce_mean(value_loss)

            loss = policy_loss + 0.25 * value_loss

        network = Network(
            s=graph_s,
            a=graph_action,
            r=graph_r,
            a_softmax=a_softmax,
            graph_v=graph_v,
            policy_loss=policy_loss,
            value_loss=value_loss,
            policy_entropy=policy_entropy,
            advantage=advantage,
            loss=loss)

        return network
