from collections import deque

import easy_tf_log
import numpy as np

import utils
from multi_scope_train_op import *
from network import create_network

G = 0.99


class Worker:

    def __init__(self, sess, env, worker_n, log_dir, debug, optimizer):
        self.sess = sess
        self.env = env

        self.worker_n = worker_n
        worker_scope = "worker_{}".format(worker_n)
        self.network = create_network(scope=worker_scope, debug=debug,
                                      n_actions=env.action_space.n)
        self.summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        self.scope = worker_scope

        self.logger = easy_tf_log.Logger()
        self.logger.set_writer(self.summary_writer.event_writer)

        self.train_op, grads_norm = create_train_op(
            self.network.loss,
            optimizer,
            compute_scope=worker_scope,
            apply_scope='global',
            max_grad_norm=0.5)

        grads_norm_policy = tf.global_norm(
            tf.gradients(self.network.policy_loss, tf.trainable_variables()))
        grads_norm_value = tf.global_norm(
            tf.gradients(self.network.value_loss, tf.trainable_variables()))

        # TODO
        utils.add_rmsprop_monitoring_ops(optimizer, 'combined_loss')

        tf.summary.scalar
        log_name_vals = [
            ('rl/value_loss', self.network.value_loss),
            ('rl/policy_loss', self.network.policy_loss),
            ('rl/combined_loss', self.network.loss),
            ('rl/policy_entropy', self.network.policy_entropy),
            ('rl/advantage_mean', tf.reduce_mean(self.network.advantage)),
            ('gradients/norm', grads_norm),
            ('gradients/norm_policy', grads_norm_policy),
            ('gradients/norm_value', grads_norm_value),
        ]
        summaries = []
        for name, val in log_name_vals:
            summary = tf.summary.scalar("worker_{}/".format(worker_n) + name,
                                        val)
            summaries.append(summary)
        self.summary_ops = tf.summary.merge(summaries)

        self.copy_ops = utils.create_copy_ops(from_scope='global',
                                              to_scope=self.scope)

        self.steps = 0
        self.render = False

        self.value_log = deque(maxlen=100)
        self.fig = None

        self.last_o = self.env.reset()

        self.episode_values = []

    def sync_scopes(self):
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

    def logkv(self, key, value):
        self.logger.logkv("worker_{}/".format(self.worker_n) + key, value)

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
            self.episode_values.append(v)

            self.last_o, r, done, _ = self.env.step(a)

            # The state used to choose the action.
            # Not the current state. The previous state.
            states.append(np.copy(s))
            actions.append(a)
            rewards.append(r)

            if self.render:
                self.env.render()
                self.value_log.append(v)
                self.value_graph()

            if done:
                break

        self.logkv('rl/batch_reward_sum', sum(rewards))

        last_state = np.copy(self.last_o)

        if done:
            returns = utils.rewards_to_discounted_returns(rewards, G)
            self.last_o = self.env.reset()
            episode_value_sum = sum(self.episode_values)
            episode_value_mean = episode_value_sum / len(self.episode_values)
            self.logkv('rl/episode_value_sum', episode_value_sum)
            self.logkv('rl/episode_value_mean', episode_value_mean)
            self.episode_values = []
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
        summaries, _ = self.sess.run([self.summary_ops,
                                      self.train_op],
                                     feed_dict)
        self.summary_writer.add_summary(summaries, self.steps)

        self.steps += 1

        return len(states)
