import re

import tensorflow as tf


# TODO: don't create gradient buffers for None variables

def strip_var_name(name):
    """
    e.g. scope/weights:0 -> weights
    """
    return re.match('\w*/([^:]*):\w*', name).group(1)


def create_train_ops(loss, optimizer, update_scope, apply_scope):
    """
    - For each trainable variable:
    -  Create gradient operator
    -  Create a gradient buffer
    -  Create an update operator which adds the gradient
       to the gradient buffer
    -  Create an operator which will apply the gradient buffer

    update_scope: the scope in which to calculate gradients
    apply_scope: the scope in which to apply the gradient buffers
    """

    # Create a dictionary mapping from variable name to
    # gradients calculated in update_scope
    update_tvs = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=update_scope)
    grads_and_vars = optimizer.compute_gradients(loss, update_tvs)
    grads_dict = {}
    for grads, var in grads_and_vars:
        var_name = strip_var_name(var.name)
        grads_dict[var_name] = grads

    # Discard variables which don't have gradients
    grads_dict = {v: g for v, g in grads_dict.items() if g is not None}

    # Create gradient buffers (indexed by variable name)
    grad_bufs = {}
    for var_name, grads in grads_dict.items():
        name = "grad_buf_%s" % var_name
        grad_bufs[var_name] = \
            tf.Variable(tf.zeros(shape=grads.get_shape()),
                        trainable=False, name=name)

    # Create ops which add the computed gradients
    # to the gradient buffers
    update_ops = []
    for var_name, grad in grads_dict.items():
        update_ops.append(tf.assign_add(grad_bufs[var_name], grad))
    update_gradients = tf.group(*update_ops)

    # Create a dictionary mapping from variable names to
    # variables in apply_scope
    apply_tvs = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=apply_scope)
    apply_tvs_dict = {}
    for var in apply_tvs:
        var_name = strip_var_name(var.name)
        apply_tvs_dict[var_name] = var

    # Create an operator which applies the gradient buffers
    # to apply_scope
    grad_bufs_and_vars = []
    for var_name, grad_buf in grad_bufs.items():
        grad_bufs_and_vars.append((grad_buf, apply_tvs_dict[var_name]))
    apply_gradients = optimizer.apply_gradients(grad_bufs_and_vars)

    # Create an operator which zeros out the buffers
    zero_ops = []
    for grad_buf in grad_bufs.values():
        op = tf.assign(grad_buf, tf.zeros(shape=grad_buf.get_shape()))
        zero_ops.append(op)
    zero_gradients = tf.group(*zero_ops)

    return update_gradients, apply_gradients, zero_gradients, grad_bufs
