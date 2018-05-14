import re

import tensorflow as tf


def strip_var_name(name):
    """
    e.g. scope/weights:0 -> weights
    """
    return re.match('\w*/([^:]*):\w*', name).group(1)


def create_train_op(compute_scope_loss, optimizer, compute_scope, apply_scope):
    """
    compute_scope: the scope in which to calculate gradients
    apply_scope: the scope in which to apply the gradients
    """

    # Create a dictionary mapping from variable name to
    # gradients calculated in compute_scope
    compute_tvs = tf.trainable_variables(compute_scope)
    grads_and_compute_scope_vars = optimizer.compute_gradients(compute_scope_loss,
                                                               compute_tvs)
    compute_scope_grads_dict = {}
    for grad, var in grads_and_compute_scope_vars:
        if grad is None:
            continue
        var_name = strip_var_name(var.name)
        compute_scope_grads_dict[var_name] = grad

    grads_norm = tf.global_norm(list(compute_scope_grads_dict.values()))

    # Create a dictionary mapping from variable names to
    # variables in apply_scope
    apply_tvs = tf.trainable_variables(apply_scope)
    apply_tvs_dict = {}
    for var in apply_tvs:
        var_name = strip_var_name(var.name)
        apply_tvs_dict[var_name] = var

    # Create an operator which applies gradients to variables in apply_scope
    grads_and_compute_scope_vars = []
    for var_name, grad in compute_scope_grads_dict.items():
        grads_and_compute_scope_vars.append((grad, apply_tvs_dict[var_name]))
    train_op = optimizer.apply_gradients(grads_and_compute_scope_vars)

    return train_op, grads_norm
