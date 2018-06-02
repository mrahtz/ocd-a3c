import easy_tf_log
import numpy as np

import utils
from multi_scope_train_op import *
from params import DISCOUNT_FACTOR


class Worker:

    def __init__(self, sess, env, network, log_dir):
        self.sess = sess
        self.env = env
        self.network = network

        if network.summaries_op is not None:
            self.summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
            self.logger = easy_tf_log.Logger()
            self.logger.set_writer(self.summary_writer.event_writer)
        else:
            self.summary_writer = None
            self.logger = None

        self.updates = 0
        self.last_obs = self.env.reset()
        self.episode_values = []

    def run_update(self, n_steps):
        self.sess.run(self.network.sync_with_global_ops)

        actions, done, rewards, states = self.run_steps(n_steps)
        returns = self.calculate_returns(done, rewards)

        if done:
            self.last_obs = self.env.reset()
            episode_value_sum = sum(self.episode_values)
            episode_value_mean = episode_value_sum / len(self.episode_values)
            if self.logger:
                self.logger.logkv('rl/episode_value_sum', episode_value_sum)
                self.logger.logkv('rl/episode_value_mean', episode_value_mean)
            self.episode_values = []

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

    def calculate_returns(self, done, rewards):
        if done:
            returns = utils.rewards_to_discounted_returns(rewards,
                                                          DISCOUNT_FACTOR)
        else:
            # If we're ending in a non-terminal state, in order to calculate
            # returns, we need to know the return of the final state.
            # We estimate this using the value network.
            s = np.moveaxis(self.last_obs, source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            last_value = self.sess.run(self.network.graph_v,
                                       feed_dict=feed_dict)[0]
            rewards += [last_value]
            returns = utils.rewards_to_discounted_returns(rewards,
                                                          DISCOUNT_FACTOR)
            returns = returns[:-1]  # Chop off last_value
        return returns

    def run_steps(self, n_steps):
        states = []
        actions = []
        rewards = []

        for _ in range(n_steps):
            s = np.moveaxis(self.last_obs, source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            [a_p], [v] = self.sess.run([self.network.a_softmax,
                                        self.network.graph_v],
                                       feed_dict=feed_dict)
            a = np.random.choice(self.env.action_space.n, p=a_p)
            self.episode_values.append(v)

            self.last_obs, r, done, _ = self.env.step(a)

            # The state used to choose the last action.
            # Not the current state. The previous state.
            states.append(np.copy(s))
            actions.append(a)
            rewards.append(r)

            if done:
                break

        return actions, done, rewards, states
