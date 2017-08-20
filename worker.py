from collections import deque
import numpy as np
import tensorflow as tf

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

        self.reward_var = tf.Variable(0.0)
        self.reward_summary = tf.summary.scalar('reward', self.reward_var)
        self.smoothed_reward = None

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

        if self.smoothed_reward is None:
            self.smoothed_reward = reward_sum
        else:
            self.smoothed_reward = self.smoothed_reward * 0.99 + reward_sum * 0.01
        print("Smoothed reward sum is %.1f" % self.smoothed_reward)

        self.sess.run(tf.assign(self.reward_var, self.smoothed_reward))
        summ = self.sess.run(self.reward_summary)
        self.summary_writer.add_summary(summ, self.steps)

    def sync_network(self):
        copy_network(self.sess,
                     from_scope='global',
                     to_scope=self.scope)

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

            if r != 0:
                print("Got reward", r)
            self.frame_stack.append(o)
            self.episode_rewards.append(r)
            list_set(rewards, i, r)
            list_set(states, i + 1, np.copy(self.frame_stack))

            i += 1

        if done:
            print("Episode done!")
            r = 0
        else:
            # We're not at the end of an episode, so we have to estimate
            # the value of the current state using the value network
            s = np.moveaxis(states[i], source=0, destination=-1) # the last state
            feed_dict = {self.network.s: [s]}
            r = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]

        # i - 1 to 0
        # (Why start from i - 1, rather than i?
        #  So that we miss out the last state.)
        for j in reversed(range(i)):
            s = np.moveaxis(states[j], source=0, destination=-1)

            if rewards[j] != 0:
                r = rewards[j]
            else:
                r = rewards[j] + G * r
            feed_dict = {self.network.s: [s]}
            v = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]
            advantage = r - v

            feed_dict = {self.network.s: [s],
                         self.network.a: [actions[j] - 1], # map from possible actions (1, 2, 3) -> (0, 1, 2)
                         self.network.r: [advantage]}
            self.sess.run([self.update_policy_gradients,
                      self.update_value_gradients],
                      feed_dict)
        self.sess.run([self.apply_policy_gradients,
                       self.apply_value_gradients])
        self.sess.run([self.zero_policy_gradients,
                       self.zero_value_gradients])

        if done:
            self.log_rewards()
            self.episode_rewards = []

        self.steps += 1

        return done
