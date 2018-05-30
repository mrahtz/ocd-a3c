import multiprocessing
import os.path as osp
import queue
import random
import socket
import subprocess
import time
from multiprocessing import Queue, Pipe, Process
from threading import Thread

import numpy as np
import tensorflow as tf


def get_port_range(start_port, n_ports, random_stagger=False):
    # If multiple runs try and call this function at the same time,
    # the function could return the same port range.
    # To guard against this, automatically offset the port range.
    if random_stagger:
        start_port += random.randint(0, 20) * n_ports

    free_range_found = False
    while not free_range_found:
        ports = []
        for port_n in range(n_ports):
            port = start_port + port_n
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                ports.append(port)
            except socket.error as e:
                if e.errno == 98 or e.errno == 48:
                    print("Warning: port {} already in use".format(port))
                    break
                else:
                    raise e
            finally:
                s.close()
        if len(ports) < n_ports:
            # The last port we tried was in use
            # Try again, starting from the next port
            start_port = port + 1
        else:
            free_range_found = True

    return ports


def with_prob(p):
    if np.random.random() < p:
        return True
    else:
        return False


def rewards_to_discounted_returns(rewards, discount_factor):
    returns = np.zeros_like(rewards, dtype=np.float32)
    returns[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        returns[i] = rewards[i] + discount_factor * returns[i + 1]
    return returns


def logit_entropy(logits):
    """
    Numerically-stable entropy directly from logits.

    We want to calculate p = exp(logits) / sum(exp(logits)),
    then do -sum(p * log(p)).

    There are two things we need to be careful of:
    - If one of the logits is large, exp(logits) will overflow.
    - If one of the probabilities is zero, we'll accidentally do log(0).
      (Entropy /is/ still well-defined if one of the probabilities is zero.
      we just miss out that probability from the sum.)
    
    The first problem is just a matter of using a numerically-stable softmax.

    For the second problem, if we have access to the logits, there's a trick we
    can use. Note that if we're calculating probabilities from logits, none of
    the probabilities should ever be zero. If we do up with a zero probability,
    it's only because of rounding. To get around this, when computing log(p),
    we don't compute probabilities explicitly, but instead compute the result
    directly in terms of the logits. For example:
      logits = [0, 1000]
      log(probs) = log(exp(logits)/sum(exp(logits)))
                 = log(exp(logits)) - log(sum(exp(logits)))
                 = logits - log_sum(exp(logits))
                 = [0, 1000] - log(exp(0) + exp(1000))
                 = [0, 1000] - log(1 + exp(1000))
                 = [0, 1000] - log(exp(1000))               (approximately)
                 = [0, 1000] - 1000
                 = [-1000, 0]
    """
    # We support either:
    # - 1D list of logits
    # - A 2D list, batch size x logits
    assert len(logits.shape) <= 2
    # keepdims=True is necessary so that we get a result which is
    # batch size x 1 instead of just batch size
    logp = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    nlogp = -logp
    probs = tf.nn.softmax(logits, axis=-1)
    nplogp = probs * nlogp
    # This reduce_sum is just the final part of the entropy calculation.
    # Don't worry - we return the entropy for each individual item in the batch.
    return tf.reduce_sum(nplogp, axis=-1, keepdims=True)


def create_copy_ops(from_scope, to_scope):
    """
    Create operations to mirror the values from all trainable variables
    in from_scope to to_scope.
    """
    from_tvs = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_tvs = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)

    from_dict = {var.name: var for var in from_tvs}
    to_dict = {var.name: var for var in to_tvs}
    copy_ops = []
    for to_name, to_var in to_dict.items():
        from_name = to_name.replace(to_scope, from_scope)
        from_var = from_dict[from_name]
        op = to_var.assign(from_var.value())
        copy_ops.append(op)

    return copy_ops


class MemoryProfiler:
    STOP_CMD = 0

    def __init__(self, pid, log_path):
        self.pid = pid
        self.log_path = log_path
        self.cmd_queue = Queue()
        self.t = None

    def start(self):
        self.t = Thread(target=self.profile)
        self.t.start()

    def stop(self):
        self.cmd_queue.put(self.STOP_CMD)
        self.t.join()

    def profile(self):
        import memory_profiler
        f = open(self.log_path, 'w+')
        while True:
            # 5 samples, 1 second apart
            memory_profiler.memory_usage(self.pid, stream=f,
                                         timeout=5, interval=1)
            f.flush()

            try:
                cmd = self.cmd_queue.get(timeout=0.1)
                if cmd == self.STOP_CMD:
                    f.close()
                    break
            except queue.Empty:
                pass


def get_git_rev():
    if not osp.exists('.git'):
        git_rev = "unkrev"
    else:
        git_rev = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().rstrip()
    return git_rev


def set_random_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Timer:
    """
    A simple timer class.

    Set the timer duration with the duration_seconds argument to the
    constructor.

    Start the timer by calling reset().

    Check if the timer is done by calling done().
    """

    def __init__(self, duration_seconds):
        self.duration_seconds = duration_seconds
        self.start_time = None

    def reset(self):
        self.start_time = time.time()

    def done(self):
        cur_time = time.time()
        if cur_time - self.start_time > self.duration_seconds:
            return True
        else:
            return False


class ProcessSafeCounter:

    def __init__(self):
        self.value = multiprocessing.Value('i', 0)

    def increment(self, n=1):
        with self.value.get_lock():
            self.value.value += n

    def __int__(self):
        return self.value.value

    def __repr__(self):
        return str(self.value.value)


class GraphCounter:

    def __init__(self, sess):
        self.sess = sess
        self.value = tf.Variable(0, trainable=False)
        self.increment_by = tf.placeholder(tf.int32)
        self.increment_op = self.value.assign_add(self.increment_by)

    def __int__(self):
        return int(self.sess.run(self.value))

    def increment(self, n=1):
        self.sess.run(self.increment_op,
                      feed_dict={self.increment_by: n})


class SubProcessEnv():
    @staticmethod
    def env_process(pipe, make_env_fn):
        env = make_env_fn()
        pipe.send((env.observation_space, env.action_space))
        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                action = data
                obs, reward, done, info = env.step(action)
                pipe.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs = env.reset()
                pipe.send(obs)

    def __init__(self, make_env_fn):
        p1, p2 = Pipe()
        self.pipe = p1
        self.proc = Process(target=self.env_process, args=[p2, make_env_fn])
        self.proc.start()
        self.observation_space, self.action_space = self.pipe.recv()

    def reset(self):
        self.pipe.send(('reset', None))
        return self.pipe.recv()

    def step(self, action):
        self.pipe.send(('step', action))
        return self.pipe.recv()

    def close(self):
        self.proc.terminate()


def make_grad_summaries(vars, grads):
    summaries = []
    for v, g in zip(vars, grads):
        if g is None:
            continue
        # strip "worker_0/"
        v_name = '/'.join(v.name.split('/')[1:])
        summary_name = "grads/{}".format(v_name)
        histogram = tf.summary.histogram(summary_name, g)
        summaries.append(histogram)
    return summaries


def make_histograms(tensors, name):
    summaries = []
    for tensor in tensors:
        # strip "worker_0/"; extract a nice name
        tensor_name = tensor.name.split('/')
        if name == 'activations':
            tensor_name = [tensor_name[1]]
        elif name == 'rms':
            tensor_name = tensor_name[1:3]
        else:
            tensor_name = tensor_name[1:]
        tensor_name = '/'.join(tensor_name)

        summary_name = "{}/{}".format(name, tensor_name)
        histogram = tf.summary.histogram(summary_name, tensor)
        summaries.append(histogram)
    return summaries


def make_rmsprop_summaries(rmsprop_optimizer):
    rms_vars = [rmsprop_optimizer.get_slot(var, 'rms')
                for var in tf.trainable_variables()]
    rms_vars = [v for v in rms_vars if v is not None]
    summaries = make_histograms(rms_vars, 'rms')
    return summaries
