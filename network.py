from math import sqrt

import tensorflow as tf

from utils import logit_entropy


class Network:

    def __init__(self, scope, n_actions, debug=False, entropy_bonus=0.01,
                 value_loss_coef=0.25, weight_inits='ortho'):

        with tf.variable_scope(scope):
            observations = tf.placeholder(tf.float32, [None, 84, 84, 4])
            actions = tf.placeholder(tf.int64, [None])
            returns = tf.placeholder(tf.float32, [None])

            if weight_inits == 'ortho':
                kernel_initializer = tf.orthogonal_initializer(gain=sqrt(2))
            elif weight_inits == 'glorot':
                kernel_initializer = None
            conv1 = tf.layers.conv2d(
                name='conv1',
                inputs=observations,
                filters=32,
                kernel_size=8,
                strides=4,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer)

            if debug:
                # Dump observations as fed into the network to stderr,
                # for viewing with show_observations.py.
                conv1 = tf.Print(conv1, [observations],
                                 message='\ndebug observations:',
                                 # max no. of values to display; max int32
                                 summarize=2147483647)

            if weight_inits == 'ortho':
                kernel_initializer = tf.orthogonal_initializer(gain=sqrt(2))
            elif weight_inits == 'glorot':
                kernel_initializer = None
            conv2 = tf.layers.conv2d(
                name='conv2',
                inputs=conv1,
                filters=64,
                kernel_size=4,
                strides=2,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer)

            if weight_inits == 'ortho':
                kernel_initializer = tf.orthogonal_initializer(gain=sqrt(2))
            elif weight_inits == 'glorot':
                kernel_initializer = None
            conv3 = tf.layers.conv2d(
                name='conv3',
                inputs=conv2,
                filters=64,
                kernel_size=3,
                strides=1,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer)

            w, h, f = conv3.get_shape()[1:]
            conv3_unwrapped = tf.reshape(conv3, [-1, int(w * h * f)])

            if weight_inits == 'ortho':
                kernel_initializer = tf.orthogonal_initializer(gain=sqrt(2))
            elif weight_inits == 'glorot':
                kernel_initializer = None
            features = tf.layers.dense(
                name='features',
                inputs=conv3_unwrapped,
                units=512,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer)

            if weight_inits == 'ortho':
                kernel_initializer = tf.orthogonal_initializer(gain=sqrt(0.01))
            elif weight_inits == 'glorot':
                kernel_initializer = None
            a_logits = tf.layers.dense(
                name='action_logits',
                inputs=features,
                units=n_actions,
                activation=None,
                kernel_initializer=kernel_initializer)

            a_softmax = tf.nn.softmax(a_logits)

            # For the policy loss, we want to calculate log π(action_t | state_t).
            # That means we want log(action_prob_0 | state_t) if action_t = 0,
            #                    log(action_prob_1 | state_t) if action_t = 1, etc.
            # It turns out that's exactly what a cross-entropy loss gives us!
            # The cross-entropy of a distribution p wrt a distribution q is:
            #   - sum over x: p(x) * log2(q(x))
            # Note that for a categorical distribution, considering the
            # cross-entropy of the ground truth distribution wrt the
            # distribution of predicted class probabilities, p(x) is 1 if the
            # ground truth label is x and 0 otherwise. We therefore have:
            #   - log2(q(0)) if ground truth label = 0,
            #   - log2(q(1)) if ground truth label = 1, etc.
            # So here, by taking the cross-entropy of the distribution of
            # action 'labels' wrt the produced action probabilities, we can get
            # exactly what we want :)
            neglogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=a_logits, labels=actions)

            if debug:
                neglogprob = tf.Print(neglogprob, [actions],
                                      message='\ndebug actions:',
                                      summarize=2147483647)

            if weight_inits == 'ortho':
                kernel_initializer = tf.orthogonal_initializer()
            elif weight_inits == 'glorot':
                kernel_initializer = None
            graph_v = tf.layers.dense(
                name='value',
                inputs=features,
                units=1,
                activation=None,
                kernel_initializer=kernel_initializer)
            # Shape is currently (?, 1)
            # Convert to just (?)
            graph_v = graph_v[:, 0]

            advantage = returns - graph_v

            if debug:
                advantage = tf.Print(advantage, [returns],
                                     message='\ndebug returns:',
                                     summarize=2147483647)

            check_nlp = tf.assert_rank(neglogprob, 1)
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
                policy_loss = neglogprob * tf.stop_gradient(advantage)
                policy_loss = tf.reduce_mean(policy_loss)

                policy_entropy = tf.reduce_mean(logit_entropy(a_logits))
                # We want to maximise entropy, which is the same as
                # minimising negative entropy
                policy_loss -= entropy_bonus * policy_entropy

                value_loss = 0.5 * advantage ** 2
                value_loss = tf.reduce_mean(value_loss)
                value_loss *= value_loss_coef

                loss = policy_loss + value_loss

            layers = [conv1, conv2, conv3, features]

            self.s = observations
            self.a = actions
            self.r = returns
            self.a_softmax = a_softmax
            self.graph_v = graph_v
            self.policy_loss = policy_loss
            self.value_loss = value_loss
            self.policy_entropy = policy_entropy
            self.advantage = advantage
            self.loss = loss
            self.layers = layers
