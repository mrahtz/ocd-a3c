import tensorflow as tf
import numpy as np
import scipy.misc


def copy_network(sess, from_scope, to_scope):
    # TODO: only trainable variables?
    from_tvs = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope=to_scope)

    from_dict = {var.name: var for var in from_tvs}
    to_dict = {var.name: var for var in to_tvs}
    copy_ops = []
    for to_name, to_var in to_dict.items():
        op = to_var.assign(
            from_dict[to_name.replace(to_scope, from_scope)].value())
        copy_ops.append(op)
    sess.run(copy_ops)


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
