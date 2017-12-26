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

        self.val_summ = tf.summary.scalar('value_loss', self.network.value_loss)

        self.init_copy_ops()

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
        self.summary_writer.add_summary(summ, self.episode_n)


    def init_copy_ops(self):
        from_tvs = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
        to_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=self.scope)

        from_dict = {var.name: var for var in from_tvs}
        to_dict = {var.name: var for var in to_tvs}
        copy_ops = []
        for to_name, to_var in to_dict.items():
            from_name = to_name.replace(self.scope, 'global')
            from_var = from_dict[from_name]
            op = to_var.assign(from_var.value())
            copy_ops.append(op)

        self.copy_ops = copy_ops


    def sync_network(self):
        self.sess.run(self.copy_ops)


    def value_graph(self):
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
            print("Episode %d finished" % self.episode_n)
            self.log_rewards()
            self.episode_rewards = []
            self.episode_n += 1

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

            feed_dict = {self.network.s: [s],
                         # map from possible actions (1, 2, 3) -> (0, 1, 2)
                         self.network.a: [a],
                         self.network.r: [r]}

            self.sess.run([self.update_policy_gradients,
                           self.update_value_gradients],
                          feed_dict)

        feed_dict = {self.network.s: s_batch,
                     self.network.a: a_batch,
                     self.network.r: r_batch}
        val_loss = self.sess.run(self.val_summ, feed_dict)
        self.summary_writer.add_summary(val_loss, self.steps)

        self.sess.run([self.apply_policy_gradients,
                       self.apply_value_gradients])
        self.sess.run([self.zero_policy_gradients,
                       self.zero_value_gradients])

        self.steps += 1

        return done
