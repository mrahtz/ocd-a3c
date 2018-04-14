import random
import socket

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


def rewards_to_discounted_returns(r, discount_factor):
    returns = np.zeros_like(np.array(r), dtype=np.float32)
    returns[-1] = r[-1]
    for i in range(len(r) - 2, -1, -1):
        returns[i] = r[i] + discount_factor * returns[i + 1]
    return returns


def entropy(logits, dims=-1):
    """
    Numerically-stable entropy.
    From https://gist.github.com/vahidk/5445ce374a27f6d452a43efb1571ea75.
    """
    probs = tf.nn.softmax(logits, dims)
    nplogp = probs * (tf.reduce_logsumexp(logits, dims, keepdims=True) - logits)
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
