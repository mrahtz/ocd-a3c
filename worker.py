from collections import deque
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gym

from network import create_network
from train_ops import *
from utils import *

G = 0.99
N_ACTIONS = 3
ACTIONS = np.arange(N_ACTIONS) + 1
N_FRAMES_STACKED = 4
N_MAX_NOOPS = 30

def list_set(l, i, val):
    assert(len(l) == i)
    l.append(val)

class Worker:

    def __init__(self, sess, worker_n, env_name, summary_writer):
        self.sess = sess
        self.env = EnvWrapper(gym.make(env_name), prepro2=prepro2, frameskip=4)

        worker_scope = "worker_%d" % worker_n
        self.network = create_network(worker_scope)
        self.summary_writer = summary_writer
        self.scope = worker_scope

        self.reward = tf.Variable(0.0)
        self.reward_summary = tf.summary.scalar('reward', self.reward)

        policy_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)

        self.update_policy_gradients, self.apply_policy_gradients, self.zero_policy_gradients, self.grad_bufs_policy = \
            create_train_ops(self.network.policy_loss,
                             policy_optimizer,
                             update_scope=worker_scope,
                             apply_scope='global')

        self.update_value_gradients, self.apply_value_gradients, self.zero_value_gradients, self.grad_bufs_value = \
            create_train_ops(self.network.value_loss,
                             value_optimizer,
                             update_scope=worker_scope,
                             apply_scope='global')

        self.frame_stack = deque(maxlen=N_FRAMES_STACKED)
        self.reset_env()

        self.t_max = 10000
        self.steps = 0
        self.episode_rewards = []
        self.render = False

        self.value_log = deque(maxlen=100)
        self.fig = None

    def reset_env(self):
        self.env.reset()
        n_noops = np.random.randint(low=0, high=N_MAX_NOOPS+1)
        print("%d no-ops..." % n_noops)
        for i in range(n_noops):
            o, _, _, _ = self.env.step(0)
            self.frame_stack.append(o)
        while len(self.frame_stack) < N_FRAMES_STACKED:
            print("One more...")
            o, _, _, _ = self.env.step(0)
            self.frame_stack.append(o)
        print("No-ops done")

    def log_rewards(self):
        reward_sum = sum(self.episode_rewards)
        print("Reward sum was", reward_sum)
        self.sess.run(tf.assign(self.reward, reward_sum))
        summ = self.sess.run(self.reward_summary)
        self.summary_writer.add_summary(summ, self.steps)

    def sync_network(self):
        copy_network(self.sess,
                     from_scope='global',
                     to_scope=self.scope)

    def value_graph(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches(2, 2)
            maxlen = 100
            self.ax.set_xlim([0, maxlen])
            self.ax.set_ylim([0, 1])
            self.line, = self.ax.plot([], [])

            self.fig.show()
            self.fig.canvas.draw()
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.fig.canvas.restore_region(self.bg)

        ydata = list(self.value_log)
        xdata = list(range(len(self.value_log)))
        if max(ydata) > self.ax.get_ylim()[1]:
            self.ax.set_ylim([0, max(ydata)])
        self.line.set_data(xdata, ydata)

        self.ax.draw_artist(self.line)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()

    def run_step(self):
        states = []
        actions = []
        rewards = []
        i = 0

        self.sess.run([self.zero_policy_gradients,
                  self.zero_value_gradients])
        self.sync_network()

        list_set(states, i, self.frame_stack)

        done = False
        while not done and i < self.t_max:
            #print("Step %d" % i)
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

            if r != 0:
                print("Got reward", r)
            self.frame_stack.append(o)
            self.episode_rewards.append(r)
            list_set(rewards, i, r)
            list_set(states, i + 1, np.copy(self.frame_stack))

            i += 1

        if done:
            print("Episode done")
            self.log_rewards()
            self.episode_rewards = []

        # Calculate initial value for R
        if done:
            # Terminal state
            r = 0
        else:
            # Non-terminal state
            # Estimate the value of the current state using the value network
            # (states[i]: the last state)
            s = np.moveaxis(states[i], source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            r = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]

        # i - 1 to 0
        # (Why start from i - 1, rather than i?
        #  So that we miss out the last state.)
        for j in reversed(range(i)):
            s = np.moveaxis(states[j], source=0, destination=-1)
            r = rewards[j] + G * r
            feed_dict = {self.network.s: [s],
                         # map from possible actions (1, 2, 3) -> (0, 1, 2)
                         self.network.a: [actions[j] - 1], 
                         self.network.r: [r]}

            self.sess.run([self.update_policy_gradients,
                      self.update_value_gradients],
                      feed_dict)

        self.sess.run([self.apply_policy_gradients,
                       self.apply_value_gradients])
        self.sess.run([self.zero_policy_gradients,
                       self.zero_value_gradients])

        self.steps += 1

        return done
