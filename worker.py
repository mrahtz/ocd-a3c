from collections import deque

import gym
import numpy as np
from easy_tf_log import tflog

import utils
from network import create_network
from train_ops import *

G = 0.99
N_ACTIONS = 3
ACTIONS = np.arange(N_ACTIONS) + 1
N_FRAMES_STACKED = 4
N_MAX_NOOPS = 30


def list_set(l, i, val):
    assert (len(l) == i)
    l.append(val)


class Worker:

    def __init__(self, sess, env_name, worker_n, global_seed, summary_writer):
        utils.set_random_seeds(global_seed + worker_n)

        self.env = utils.EnvWrapper(gym.make(env_name),
                                    prepro2=utils.prepro2,
                                    frameskip=4)
        self.env.seed(global_seed + worker_n)

        self.sess = sess

        worker_scope = "worker_%d" % worker_n
        self.network = create_network(worker_scope)
        self.summary_writer = summary_writer
        self.scope = worker_scope

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)

        self.update_gradients, self.apply_gradients, self.zero_gradients, self.grad_bufs = \
            create_train_ops(self.network.loss,
                             optimizer,
                             update_scope=worker_scope,
                             apply_scope='global')

        tf.summary.scalar('value_loss',
                          self.network.value_loss)
        tf.summary.scalar('policy_entropy',
                          tf.reduce_mean(self.network.policy_entropy))
        self.summary_ops = tf.summary.merge_all()

        self.copy_ops = utils.create_copy_ops(from_scope='global',
                                              to_scope=self.scope)

        self.frame_stack = deque(maxlen=N_FRAMES_STACKED)
        self.reset_env()

        self.t_max = 10000
        self.steps = 0
        self.episode_rewards = []
        self.render = False
        self.episode_n = 1

        self.value_log = deque(maxlen=100)
        self.fig = None

    def reset_env(self):
        self.frame_stack.clear()
        self.env.reset()

        n_noops = np.random.randint(low=0, high=N_MAX_NOOPS + 1)
        print("%d no-ops..." % n_noops)
        for i in range(n_noops):
            o, _, _, _ = self.env.step(0)
            self.frame_stack.append(o)
        while len(self.frame_stack) < N_FRAMES_STACKED:
            print("One more...")
            o, _, _, _ = self.env.step(0)
            self.frame_stack.append(o)
        print("No-ops done")

    @staticmethod
    def log_rewards(episode_rewards):
        reward_sum = sum(episode_rewards)
        print("Reward sum was", reward_sum)
        tflog('episode_reward', reward_sum)

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

    def run_update(self):
        states = []
        actions = []
        rewards = []
        i = 0

        self.sess.run(self.zero_gradients)
        self.sync_network()

        list_set(states, i, self.frame_stack)

        done = False
        while not done and i < self.t_max:
            s = np.moveaxis(self.frame_stack, source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            a_p = self.sess.run(self.network.a_softmax, feed_dict=feed_dict)[0]
            a = np.random.choice(ACTIONS, p=a_p)
            list_set(actions, i, a)

            o, r, done, _ = self.env.step(a)

            if self.render:
                self.env.render()
                feed_dict = {self.network.s: [s]}
                v = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]
                self.value_log.append(v)
                self.value_graph()

            self.frame_stack.append(o)
            self.episode_rewards.append(r)
            list_set(rewards, i, r)
            list_set(states, i + 1, np.copy(self.frame_stack))

            i += 1

        if done:
            print("Episode %d finished" % self.episode_n)
            self.log_rewards(self.episode_rewards)
            self.episode_rewards = []
            self.episode_n += 1

        # Calculate initial value for R
        if done:
            # Terminal state
            r = 0
        else:
            # Non-terminal state
            # Estimate the value of the current state using the value network
            # (states[i]: the last state)
            s = np.moveaxis(states[i], source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            r = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]

        s_batch = []
        a_batch = []
        r_batch = []
        # i - 1 to 0
        # (Why start from i - 1, rather than i?
        #  So that we miss out the last state.)
        for j in reversed(range(i)):
            s = np.moveaxis(states[j], source=0, destination=-1)
            a = actions[j] - 1
            r = rewards[j] + G * r
            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)

        feed_dict = {self.network.s: s_batch,
                     self.network.a: a_batch,
                     self.network.r: r_batch}
        summaries, _ = self.sess.run([self.summary_ops,
                                      self.update_gradients],
                                     feed_dict)
        self.summary_writer.add_summary(summaries, self.steps)

        self.sess.run(self.apply_gradients)
        self.sess.run(self.zero_gradients)

        self.steps += 1

        return i, done
