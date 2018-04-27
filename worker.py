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
N_MAX_NOOPS = 30


def list_set(l, i, val):
    assert (len(l) == i)
    l.append(val)


class Worker:

    def __init__(self, sess, env_id, preprocess_wrapper, worker_n, seed, log_dir):
        env = gym.make(env_id)
        env.seed(seed)
        self.env = preprocess_wrapper(env)

        self.sess = sess

        worker_scope = "worker_%d" % worker_n
        self.network = create_network(worker_scope)
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
        policy_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4,
                                                     decay=0.99, epsilon=1e-5)

        self.update_gradients, self.apply_gradients, self.zero_gradients, self.grad_bufs, grads_norm = \
            create_train_ops(self.network.loss,
                             policy_optimizer,
                             update_scope=worker_scope,
                             apply_scope='global')

        rms_vars = [policy_optimizer.get_slot(var, 'rms')
                           for var in tf.trainable_variables()]
        rms_vars = [v for v in rms_vars if v is not None]
        rms_max = tf.reduce_max([tf.reduce_max(v)
                                        for v in rms_vars])
        rms_min = tf.reduce_min([tf.reduce_min(v)
                                        for v in rms_vars])
        rms_avg = tf.reduce_mean([tf.reduce_mean(v)
                                         for v in rms_vars])

        tf.summary.scalar('rms_max', rms_max)
        tf.summary.scalar('rms_min', rms_min)
        tf.summary.scalar('rms_avg', rms_avg)


        tf.summary.scalar('value_loss',
                          self.network.value_loss)
        tf.summary.scalar('policy_entropy',
                          self.network.policy_entropy)
        tf.summary.scalar('grads_norm', grads_norm)
        self.summary_ops = tf.summary.merge_all()

        self.copy_ops = utils.create_copy_ops(from_scope='global',
                                              to_scope=self.scope)

        self.reset_env()

        self.t_max = 5
        self.steps = 0
        self.episode_rewards = []
        self.render = False
        self.episode_n = 1

        self.value_log = deque(maxlen=100)
        self.fig = None

    def reset_env(self):
        self.last_o = self.env.reset()
        n_noops = np.random.randint(low=0, high=N_MAX_NOOPS + 1)
        print("%d no-ops..." % n_noops)
        for i in range(n_noops):
            self.last_o, _, _, _ = self.env.step(0)
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

        list_set(states, i, np.copy(self.last_o))

        done = False
        while not done and i < self.t_max:
            s = np.moveaxis(self.last_o, source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            a_p = self.sess.run(self.network.a_softmax, feed_dict=feed_dict)[0]
            a = np.random.choice(ACTIONS, p=a_p)
            list_set(actions, i, a)

            self.last_o, r, done, _ = self.env.step(a)

            if self.render:
                self.env.render()
                feed_dict = {self.network.s: [s]}
                v = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]
                self.value_log.append(v)
                self.value_graph()

            self.episode_rewards.append(r)
            list_set(rewards, i, r)
            list_set(states, i + 1, np.copy(self.last_o))

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
        summaries, _ = self.sess.run([self.summary_ops, self.update_gradients],
                                     feed_dict)
        self.summary_writer.add_summary(summaries, self.steps)

        self.sess.run(self.apply_gradients)
        self.sess.run(self.zero_gradients)

        self.steps += 1

        return i, done
