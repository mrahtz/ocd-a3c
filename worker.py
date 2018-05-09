from collections import deque

import numpy as np
from easy_tf_log import tflog

import utils
from network import create_network
from debug_wrappers import NumberFrames, MonitorEnv
from multi_scope_train_op import *

G = 0.99


class Worker:

    def __init__(self, sess, env, worker_n, log_dir, max_n_noops, debug):
        self.sess = sess
        self.env = env

        worker_scope = "worker_%d" % worker_n
        self.worker_n = worker_n
        self.network = create_network(scope=worker_scope, debug=debug,
                                      n_actions=env.action_space.n)
        self.summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        self.scope = worker_scope

        # From the paper, Section 4, Asynchronous RL Framework,
        # subsection Optimization:
        # "We investigated three different optimization algorithms in our
        #  asynchronous framework – SGD with momentum, RMSProp without shared
        #  statistics, and RMSProp with shared statistics.
        #  We used the standard non-centered RMSProp update..."
        # "A comparison on a subset of Atari 2600 games showed that a variant
        #  of RMSProp where statistics g are shared across threads is
        #  considerably more robust than the other two methods."
        #
        # TensorFlow's RMSPropOptimizer defaults to centered=False,
        # so we're good there. For shared statistics - RMSPropOptimizer's
        # gradient statistics variables are associated with the variables
        # supplied to apply_gradients(), which happen to be in the global scope
        # (see multi_scope_train_op.py). So we get shared statistics without any
        # specialeffort.
        #
        # In terms of hyperparameters:
        #
        # Learning rate: the paper actually runs a bunch of
        # different learning rates and presents results averaged over the
        # three best learning rates for each game. From the scatter plot of
        # performance for different learning rates, Figure 2, it looks like
        # 7e-4 is a safe bet which works across a variety of games.
        # TODO: 7e-4
        #
        # RMSprop hyperparameters: Section 8, Experimental Setup, says:
        # "All experiments used...RMSProp decay factor of α = 0.99."
        # There's no mention of the epsilon used. I see that OpenAI's
        # baselines implementation of A2C uses 1e-5 (https://git.io/vpCQt),
        # instead of TensorFlow's default of 1e-10. Remember, RMSprop divides
        # gradients by a factor based on recent gradient history. Epsilon is
        # added to that factor to prevent a division by zero. If epsilon is
        # too small, we'll get a very large update when the gradient history is
        # close to zero. So my speculation about why baselines uses a much
        # larger epsilon is: sometimes in RL the gradients can end up being
        # very small, and we want to limit the size of the update.
        policy_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4,
                                                     decay=0.99, epsilon=1e-5)

        self.policy_train_op, grads_policy_norm = create_train_op(
            self.network.policy_loss,
            policy_optimizer,
            compute_scope=worker_scope,
            apply_scope='global')

        utils.add_rmsprop_monitoring_ops(policy_optimizer, 'policy')

        tf.summary.scalar('rl/policy_entropy',
                          tf.reduce_mean(self.network.policy_entropy))
        tf.summary.scalar('gradients/norm_policy', grads_policy_norm)
        self.summary_ops = tf.summary.merge_all()

        self.copy_ops = utils.create_copy_ops(from_scope='global',
                                              to_scope=self.scope)

        self.steps = 0
        self.render = False
        self.max_n_noops = max_n_noops

        self.fig = None

        self.last_o = self.env.reset()


    def sync_scopes(self):
        self.sess.run(self.copy_ops)

    def run_update(self, n_steps):
        states = []
        actions = []
        rewards = []

        self.sync_scopes()

        for _ in range(n_steps):
            s = np.moveaxis(self.last_o, source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            [a_p], [v] = self.sess.run([self.network.a_softmax,
                                        self.network.graph_v],
                                       feed_dict=feed_dict)
            a = np.random.choice(self.env.action_space.n, p=a_p)

            self.last_o, r, done, _ = self.env.step(a)

            # The state used to choose the action.
            # Not the current state. The previous state.
            states.append(np.copy(s))
            actions.append(a)
            rewards.append(r)

            if self.render:
                self.env.render()

            if done:
                break

        last_state = np.copy(self.last_o)

        if done:
            returns = utils.rewards_to_discounted_returns(rewards, G)
            self.last_o = self.env.reset()
        else:
            assert("Not done?")

        feed_dict = {self.network.s: states,
                     self.network.a: actions,
                     self.network.r: returns}
        summaries, _ = self.sess.run([self.summary_ops,
                                         self.policy_train_op],
                                        feed_dict)
        self.summary_writer.add_summary(summaries, self.steps)

        self.steps += 1

        return len(states)
