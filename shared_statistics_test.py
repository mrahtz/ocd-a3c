#!/usr/bin/env python3

import unittest

import gym
import tensorflow as tf

import utils
from network import make_inference_network, Network
from preprocessing import generic_preprocess
from worker import Worker


class TestSharedStatistics(unittest.TestCase):
    """
    Let's be super duper sure that if we use the same optimizer instance to
    compute gradients in two different workers, the optimizer statistics
    are really shared.
    """

    def test_rmsprop_variables(self):
        """
        Test 1: let's look at the variables the optimizer creates to check
        there's no funny business.
        """
        sess = tf.Session()
        env = generic_preprocess(gym.make('Pong-v0'), max_n_noops=0)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4,
                                              decay=0.99, epsilon=1e-5)

        with tf.variable_scope('global'):
            make_inference_network(n_actions=env.action_space.n,
                                   weight_inits='glorot')

        network1 = Network(scope="worker_1",
                           n_actions=env.action_space.n,
                           entropy_bonus=0.01,
                           value_loss_coef=0.5,
                           weight_inits='glorot',
                           max_grad_norm=0.5,
                           optimizer=optimizer,
                           create_summary_ops=False,
                           debug=False)
        Worker(sess=sess, env=env, network=network1, worker_name="worker_1",
               log_dir='/tmp')

        vars1 = optimizer.variables()

        network2 = Network(scope="worker_2",
                           n_actions=env.action_space.n,
                           entropy_bonus=0.01,
                           value_loss_coef=0.5,
                           weight_inits='glorot',
                           max_grad_norm=0.5,
                           optimizer=optimizer,
                           create_summary_ops=False,
                           debug=False)
        Worker(sess=sess, env=env, network=network2, worker_name="worker_2",
               log_dir='/tmp')

        vars2 = optimizer.variables()

        self.assertNotEqual(id(vars1), id(vars2))

        # First, were any extra variables added when we created the second
        # optimizer, that might be indicative of a second set of statistics?
        self.assertLessEqual(vars1, vars2)
        # Second, are all the variables definitely associated with the global
        # set of parameters rather than the thead-local parameters?
        for v in vars1:
            self.assertIn('global', v.name)

    def test_shared_statistics(self):
        """
        Test 2: to be sure, let's do an end-to-end test to make sure updates
        on one worker are actually affecting updates on another worker.
        """

        # First, we'll do two a run where we deliberately reset RMSprop
        # statistics between worker 1's update and worker 2's update.
        # We'll record what the variables look like at the start, after the
        # worker 1's update, and after worker 2's update.
        vars_sum_init_1, \
        vars_sum_post_w1_update_1, \
        vars_sum_post_w2_update_1 = run_weight_test(reset_rmsprop=True)

        # We'll want to do a another run where we don't reset RMSprop
        # statistics, and check that worker 2's update is different. But
        # before we can do that, we also have to check that our results are
        # otherwise repeatable - that if we're getting different results,
        # it really is because of the lack of RMSprop reset and not because
        # of random seeding or something.
        vars_sum_init_2, \
        vars_sum_post_w1_update_2, \
        vars_sum_post_w2_update_2 = run_weight_test(reset_rmsprop=True)
        self.assertEqual(vars_sum_init_1, vars_sum_init_2)
        self.assertEqual(vars_sum_post_w1_update_1, vars_sum_post_w1_update_2)
        self.assertEqual(vars_sum_post_w2_update_1, vars_sum_post_w2_update_2)

        # OK, now we run without RMSprop statistics reset.
        vars_sum_init_3, \
        vars_sum_post_w1_update_3, \
        vars_sum_post_w2_update_3 = run_weight_test(reset_rmsprop=False)
        # The weights before any updates should be the same as before.
        self.assertEqual(vars_sum_init_2, vars_sum_init_3)
        # The weights after worker 1's update should also be the same.
        self.assertEqual(vars_sum_post_w1_update_2,
                         vars_sum_post_w1_update_3)
        # But the weights after worker 2's update should /not/ be the same.
        self.assertNotEqual(vars_sum_post_w2_update_2,
                            vars_sum_post_w2_update_3)


def get_var_sum(vars):
    return tf.reduce_sum([tf.reduce_sum(v) for v in vars])


def run_weight_test(reset_rmsprop):
    tf.reset_default_graph()
    utils.set_random_seeds(0)
    sess = tf.Session()
    env = generic_preprocess(gym.make('Pong-v0'), max_n_noops=0)
    env.seed(0)

    with tf.variable_scope('global'):
        make_inference_network(n_actions=env.action_space.n,
                               weight_inits='glorot')
    shared_variables = tf.global_variables()

    optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4,
                                          decay=0.99, epsilon=1e-5)

    network1 = Network(scope="worker_1",
                       n_actions=env.action_space.n,
                       entropy_bonus=0.01,
                       value_loss_coef=0.5,
                       weight_inits='glorot',
                       max_grad_norm=0.5,
                       optimizer=optimizer,
                       create_summary_ops=False,
                       debug=False)
    w1 = Worker(sess=sess, env=env, network=network1, worker_name="worker_1",
                log_dir='/tmp')

    network2 = Network(scope="worker_2",
                       n_actions=env.action_space.n,
                       entropy_bonus=0.01,
                       value_loss_coef=0.5,
                       weight_inits='glorot',
                       max_grad_norm=0.5,
                       optimizer=optimizer,
                       create_summary_ops=False,
                       debug=False)
    w2 = Worker(sess=sess, env=env, network=network2, worker_name="worker_2",
                log_dir='/tmp')

    rmsprop_init_ops = [v.initializer for v in optimizer.variables()]

    sess.run(tf.global_variables_initializer())

    vars_sum_init = sess.run(get_var_sum(shared_variables))
    w1.run_update(n_steps=1)
    vars_sum_post_w1_update = sess.run(get_var_sum(shared_variables))
    if reset_rmsprop:
        sess.run(rmsprop_init_ops)
    w2.run_update(n_steps=1)
    vars_sum_post_w2_update = sess.run(get_var_sum(shared_variables))

    return vars_sum_init, vars_sum_post_w1_update, vars_sum_post_w2_update


if __name__ == '__main__':
    unittest.main()
