import os.path as osp
import random
import socket
import subprocess
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import numpy as np
import scipy.misc


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


def discount_rewards(r, G):
    r2 = np.zeros_like(np.array(r).astype(np.float32))
    r2[-1] = r[-1]
    for i in range(len(r2) - 2, -1, -1):
        r2[i] = G * r2[i + 1]
    return r2


def rewards_to_returns(r, G):
    r2 = np.zeros_like(np.array(r).astype(np.float32))
    r2[-1] = r[-1]
    for i in range(len(r2) - 2, -1, -1):
        r2[i] = r[i] + G * r2[i + 1]
    return r2

# Based on Andrej's code
def prepro2(I):
    """ prepro 210x160 frame into 80x80 frame """
    I = I[34:194]  # crop
    I = I[::2, ::2]  # downsample by factor of 2
    I[I <= 0.4] = 0 # erase background
    I[I > 0.4] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float)

def prepro(o):
    o = np.mean(o, axis=2)
    o = o / 255.0
    return o

class EnvWrapper():
    def __init__(self, env, pool=False, frameskip=1, prepro2=None):
        self.env = env
        self.pool = pool
        self.prepro2 = prepro2
        # 1 = don't skip
        # 2 = skip every other frame
        self.frameskip = frameskip
        self.action_space = env.action_space
        # gym.utils.play() wants these two
        self.observation_space = env.observation_space
        self.unwrapped = env.unwrapped

    def reset(self):
        o = self.env.reset()
        self.prev_o = o
        o = prepro(o)
        if self.prepro2 is not None:
            o = self.prepro2(o)
        return o

    def step(self, a):
        i = 0
        done = False
        rs = []
        while i < self.frameskip and not done:
            o_raw, r, done, _ = self.env.step(a)
            rs.append(r)
            if not self.pool:
                o = o_raw
            else:
                # Note that the first frame to come out of this
                # _might_ be a little funny because the first prev_o
                # is the first frame after reset which at least in Pong
                # has a different colour palette (though it turns out
                # it works fine for Pong)
                o = np.maximum(o_raw, self.prev_o)
                self.prev_o = o_raw
            i += 1
        o = prepro(o)
        if self.prepro2 is not None:
            o = self.prepro2(o)
        r = sum(rs)
        info = None
        return o, r, done, info

    def render(self):
        self.env.render()

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
    return tf.reduce_sum(nplogp, axis=-1, keepdims=True)


def profile_memory(log_path, pid):
    import memory_profiler
    def profile():
        with open(log_path, 'w') as f:
            # timeout=99999 is necessary because for external processes,
            # memory_usage otherwise defaults to only returning a single sample
            # Note that even with interval=1, because memory_profiler only
            # flushes every 50 lines, we still have to wait 50 seconds before
            # updates.
            memory_profiler.memory_usage(pid, stream=f,
                                         timeout=99999, interval=1)

    p = Process(target=profile, daemon=True)
    p.start()
    return p


def get_git_rev():
    if not osp.exists('.git'):
        git_rev = "unkrev"
    else:
        git_rev = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().rstrip()
    return git_rev
