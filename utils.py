import numpy as np
import tensorflow as tf


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


def entropy(logits, dims=-1):
    """
    Numerically-stable entropy.
    From https://gist.github.com/vahidk/5445ce374a27f6d452a43efb1571ea75.
    """
    probs = tf.nn.softmax(logits, dims)
    nplogp = probs * (
                tf.reduce_logsumexp(logits, dims, keep_dims=True) - logits)
    return tf.reduce_sum(nplogp, dims, keep_dims=True)
