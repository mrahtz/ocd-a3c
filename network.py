from math import sqrt

import tensorflow as tf

import utils
from multi_scope_train_op import create_train_op
from utils import logit_entropy, make_grad_histograms, make_rmsprop_histograms, \
    make_histograms


def make_inference_network(n_actions, weight_inits, debug=False):
    observations = tf.placeholder(tf.float32, [None, 84, 84, 4])

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

    layers = [conv1, conv2, conv3, features]

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

    return observations, a_logits, a_softmax, graph_v, layers


def make_loss_ops(a_logits, graph_v, entropy_bonus, value_loss_coef, debug):
    actions = tf.placeholder(tf.int64, [None])
    returns = tf.placeholder(tf.float32, [None])

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
    _neglogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=a_logits, labels=actions)
    with tf.control_dependencies([tf.assert_rank(_neglogprob, 1)]):
        neglogprob = _neglogprob

    if debug:
        neglogprob = tf.Print(neglogprob, [actions],
                              message='\ndebug actions:',
                              summarize=2147483647)

    _advantage = returns - graph_v
    with tf.control_dependencies([tf.assert_rank(_advantage, 1)]):
        advantage = _advantage

    if debug:
        advantage = tf.Print(advantage, [returns],
                             message='\ndebug returns:',
                             summarize=2147483647)

    policy_entropy = tf.reduce_mean(logit_entropy(a_logits))

    # Note that the advantage is treated as a constant for the
    # policy network update step.
    # Note also that we're calculating advantages on-the-fly using
    # the value approximator. This might make us worry: what if we're
    # using the loss for training, and the advantages are calculated
    # /after/ training has changed the network? But for A3C, we don't
    # need to worry, because we compute the gradients seperately from
    # applying them.
    # We want to maximise entropy, which is the same as
    # minimising negative entropy.
    policy_loss = neglogprob * tf.stop_gradient(advantage)
    policy_loss = tf.reduce_mean(policy_loss) - entropy_bonus * policy_entropy
    value_loss = value_loss_coef * tf.reduce_mean(0.5 * advantage ** 2)
    loss = policy_loss + value_loss

    return actions, returns, advantage, policy_entropy, \
           policy_loss, value_loss, loss


class Network:

    def __init__(self, scope, n_actions,
                 entropy_bonus, value_loss_coef, weight_inits, max_grad_norm,
                 optimizer, create_summary_ops, debug=False):
        with tf.variable_scope(scope):
            observations, \
            a_logits, a_softmax, graph_v, \
            layers = make_inference_network(n_actions, weight_inits, debug)

            actions, returns, advantage, policy_entropy, \
            policy_loss, value_loss, loss = make_loss_ops(
                a_logits, graph_v,
                entropy_bonus, value_loss_coef, debug)

        sync_with_global_ops = utils.create_copy_ops(from_scope='global',
                                                     to_scope=scope)

        train_op, grads_norm = create_train_op(
            loss,
            optimizer,
            compute_scope=scope,
            apply_scope='global',
            max_grad_norm=max_grad_norm)

        self.s = observations
        self.a_softmax = a_softmax
        self.graph_v = graph_v
        self.layers = layers

        self.a = actions
        self.r = returns
        self.policy_entropy = policy_entropy
        self.advantage = advantage
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.loss = loss

        self.sync_with_global_ops = sync_with_global_ops
        self.optimizer = optimizer
        self.train_op = train_op
        self.grads_norm = grads_norm

        if create_summary_ops:
            self.summaries_op = self.create_summary_ops(scope)
        else:
            self.summaries_op = None

    def create_summary_ops(self, scope):
        variables = tf.trainable_variables(scope)
        grads_policy = tf.gradients(self.policy_loss, variables)
        grads_value = tf.gradients(self.value_loss, variables)
        grads_combined = tf.gradients(self.loss, variables)
        grads_norm_policy = tf.global_norm(grads_policy)
        grads_norm_value = tf.global_norm(grads_value)
        grads_norm_combined = tf.global_norm(grads_combined)

        scalar_summaries = [
            ('rl/policy_entropy', self.policy_entropy),
            ('rl/advantage_mean', tf.reduce_mean(self.advantage)),
            ('grads/loss_policy', self.policy_loss),
            ('grads/loss_value', self.value_loss),
            ('grads/loss_combined', self.loss),
            ('grads/norm_policy', grads_norm_policy),
            ('grads/norm_value', grads_norm_value),
            ('grads/norm_combined', grads_norm_combined),
            ('grads/norm_combined_clipped', self.grads_norm),
        ]
        summaries = []
        for name, val in scalar_summaries:
            summary = tf.summary.scalar(name, val)
            summaries.append(summary)

        summaries.extend(make_grad_histograms(variables, grads_combined))
        summaries.extend(make_rmsprop_histograms(self.optimizer))
        summaries.extend(make_histograms(self.layers, 'activations'))
        summaries.extend(make_histograms(variables, 'weights'))

        return tf.summary.merge(summaries)
