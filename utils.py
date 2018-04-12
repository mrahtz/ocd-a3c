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
            tf.reduce_logsumexp(logits, dims, keepdims=True) - logits)
    return tf.reduce_sum(nplogp, dims, keepdims=True)


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
