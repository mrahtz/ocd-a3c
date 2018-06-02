from collections import deque

import easy_tf_log
import numpy as np

import utils
from multi_scope_train_op import *
from params import DISCOUNT_FACTOR


class Worker:

    def __init__(self, sess, env, network, worker_name, log_dir):
        self.sess = sess
        self.env = env
        self.network = network
        self.worker_name = worker_name

        if network.summaries_op is not None:
            self.summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
            self.logger = easy_tf_log.Logger()
            self.logger.set_writer(self.summary_writer.event_writer)
        else:
            self.summary_writer = None
            self.logger = None

        self.render = False
        self.value_log = deque(maxlen=100)
        self.fig = None

        self.updates = 0
        self.last_o = self.env.reset()
        self.episode_values = []

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

        self.sess.run(self.network.sync_with_global_ops)

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

        # TODO gut more thoroughly
        # self.logkv('rl/batch_reward_sum', sum(rewards))

        last_state = np.copy(self.last_o)

        if done:
            returns = utils.rewards_to_discounted_returns(rewards,
                                                          DISCOUNT_FACTOR)
            self.last_o = self.env.reset()
            episode_value_sum = sum(self.episode_values)
            episode_value_mean = episode_value_sum / len(self.episode_values)
            if self.logger:
                self.logger.logkv('rl/episode_value_sum', episode_value_sum)
                self.logger.logkv('rl/episode_value_mean', episode_value_mean)
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
            returns = utils.rewards_to_discounted_returns(rewards,
                                                          DISCOUNT_FACTOR)
            returns = returns[:-1]  # Chop off last_value

        feed_dict = {self.network.s: states,
                     self.network.a: actions,
                     self.network.r: returns}
        self.sess.run(self.network.train_op, feed_dict)

        if self.summary_writer and \
                self.updates != 0 and self.updates % 100 == 0:
            summaries = self.sess.run(self.network.summaries_op, feed_dict)
            self.summary_writer.add_summary(summaries, self.updates)

        self.updates += 1

        return len(states)
