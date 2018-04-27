from collections import deque

import gym
import numpy as np
from easy_tf_log import tflog

import utils
from network import create_network
from debug_wrappers import NumberFrames
from train_ops import *

G = 0.99
N_ACTIONS = 3
ACTIONS = np.arange(N_ACTIONS) + 1


class Worker:

    def __init__(self, sess, env_id, preprocess_wrapper, worker_n, seed,
                 log_dir, max_n_noops, debug):
        env = gym.make(env_id)
        env.seed(seed)
        if debug:
            env = NumberFrames(env)
        self.env = preprocess_wrapper(env, max_n_noops)

        self.sess = sess

        worker_scope = "worker_%d" % worker_n
        self.worker_n = worker_n
        self.network = create_network(worker_scope, debug)
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
        # (see train_ops.py). So we get shared statistics without any special
        # effort.
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
        optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4,
                                              decay=0.99, epsilon=1e-5)

        self.update_gradients, self.apply_gradients, self.zero_gradients, self.grad_bufs, grads_norm = \
            create_train_ops(self.network.loss,
                             optimizer,
                             max_grad_norm=0.5,
                             update_scope=worker_scope,
                             apply_scope='global')

        utils.add_rmsprop_monitoring_ops(optimizer, 'loss')

        tf.summary.scalar('rl/value_loss',self.network.value_loss)
        tf.summary.scalar('rl/policy_entropy', self.network.policy_entropy)
        tf.summary.scalar('gradients/norm', grads_norm)
        self.summary_ops = tf.summary.merge_all()

        self.copy_ops = utils.create_copy_ops(from_scope='global',
                                              to_scope=self.scope)

        self.steps = 0
        self.episode_rewards = []
        self.render = False
        self.episode_n = 1
        self.max_n_noops = max_n_noops

        self.value_log = deque(maxlen=100)
        self.fig = None

        self.last_o = self.env.reset()

    def sync_network(self):
        self.sess.run(self.copy_ops)

    def value_graph(self):
        import matplotlib.pyplot as plt
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches(2, 2)
            self.ax.set_xlim([0, 100])
            self.ax.set_ylim([0, 2.0])
            self.line, = self.ax.plot([], [])

            self.fig.show()
            self.fig.canvas.draw()
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.fig.canvas.restore_region(self.bg)

        ydata = list(self.value_log)
        xdata = list(range(len(self.value_log)))
        self.line.set_data(xdata, ydata)

        self.ax.draw_artist(self.line)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()

    def run_update(self, n_steps):
        states = []
        actions = []
        rewards = []

        self.sess.run(self.zero_gradients)
        self.sync_network()

        for _ in range(n_steps):
            s = np.moveaxis(self.last_o, source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            a_p = self.sess.run(self.network.a_softmax, feed_dict=feed_dict)[0]
            a = np.random.choice(ACTIONS, p=a_p)

            self.last_o, r, done, _ = self.env.step(a)

            # The state used to choose the action.
            # Not the current state. The previous state.
            states.append(np.copy(s))
            actions.append(a - 1)
            rewards.append(r)
            self.episode_rewards.append(r)

            if self.render:
                self.env.render()
                feed_dict = {self.network.s: [s]}
                v = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]
                self.value_log.append(v)
                self.value_graph()

            if done:
                break

        last_state = np.copy(self.last_o)

        if done:
            reward_sum = sum(self.episode_rewards)
            print("Worker {} episode {} finished; reward {}".format(
                self.worker_n,
                self.episode_n,
                reward_sum)
            )
            tflog('rl/episode_reward', reward_sum)

            self.episode_rewards = []
            self.episode_n += 1

        tflog('batch_reward_sum', sum(rewards))

        if done:
            returns = utils.rewards_to_discounted_returns(rewards, G)
            self.last_o = self.env.reset()
        else:
            # If we're ending in a non-terminal state, in order to calculate
            # returns, we need to know the return of the final state.
            # We estimate this using the value network.
            s = np.moveaxis(last_state, source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            last_value = self.sess.run(self.network.graph_v,
                                       feed_dict=feed_dict)[0]
            rewards += [last_value]
            returns = utils.rewards_to_discounted_returns(rewards, G)
            returns = returns[:-1]  # Chop off last_value

        feed_dict = {self.network.s: states,
                     self.network.a: actions,
                     self.network.r: returns}
        summaries, _, _ = self.sess.run([self.summary_ops, self.update_gradients],
                                        feed_dict)
        self.summary_writer.add_summary(summaries, self.steps)

        self.sess.run(self.apply_gradients)
        self.sess.run(self.zero_gradients)

        self.steps += 1

        return len(states)
