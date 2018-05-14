#!/usr/bin/env python3

import os
import os.path as osp
import time
from multiprocessing import Process

import easy_tf_log
import gym
import tensorflow as tf

import utils
from debug_wrappers import NumberFrames, MonitorEnv
from network import create_network
from params import parse_args
from worker import Worker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def run_worker(worker_n, debug, log_dir, n_steps_to_run,
                 steps_per_update, step_counter, update_counter, env_id,
               n_workers, max_n_noops, preprocess_wrapper, seed):
    env = gym.make(env_id)
    env_seed = seed * n_workers + worker_n
    env.seed(env_seed)
    if debug:
        env = NumberFrames(env)
    env = preprocess_wrapper(env, max_n_noops)
    env = MonitorEnv(env, "worker_{}".format(worker_n))

    tf.reset_default_graph()
    sess = tf.Session()
    create_network('global', n_actions=env.action_space.n)
    optimizer = optimizer_plz()

    worker_log_dir = osp.join(log_dir, "worker_{}".format(worker_n))
    os.makedirs(worker_log_dir)
    w = Worker(sess=sess,
               env=env,
               worker_n=worker_n,
               log_dir=worker_log_dir,
               debug=debug,
               optimizer=optimizer)
    easy_tf_log.set_writer(w.summary_writer.event_writer)

    sess.run(tf.global_variables_initializer())

    while int(step_counter) < n_steps_to_run:
        steps_ran = w.run_update(steps_per_update)
        step_counter.increment(steps_ran)
        update_counter.increment(1)


def optimizer_plz():
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
    # so we're good there.
    #
    # For shared statistics, we supply the same optimizer instance to all
    # workers. Couldn't they still end up using
    # different statistics somehow?
    # a) From the source, RMSPropOptimizer's gradient statistics variables
    #    are associated with the variables supplied to apply_gradients(),
    #    which happen to be the global set of variables shared between all
    #    threads variables (see multi_scope_train_op.py).
    # b) Empirically, no. See shared_statistics_test.py.
    #
    # In terms of hyperparameters:

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

    optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4,
                                          decay=0.99, epsilon=1e-5)
    return optimizer

def main():
    args, log_dir, preprocess_wrapper, ckpt_timer = parse_args()
    easy_tf_log.set_dir(log_dir)

    step_counter = utils.MultiprocessCounter()
    update_counter = utils.MultiprocessCounter()

    def s(worker_n):
        run_worker(worker_n, args.debug, log_dir, args.n_steps,
                   args.steps_per_update, step_counter, update_counter,
                   args.env_id, args.n_workers, args.max_n_noops,
                   preprocess_wrapper, args.seed)
    worker_threads = [Process(target=s, args=[i])
                      for i in range(args.n_workers)]
    for w in worker_threads:
        w.start()

    prev_t = time.time()
    prev_steps = int(step_counter)
    while True:
        time.sleep(1.0)

        cur_t = time.time()
        cur_steps = int(step_counter)
        steps_per_second = (cur_steps - prev_steps) / (cur_t - prev_t)
        print(steps_per_second)
        easy_tf_log.tflog('misc/steps_per_second', steps_per_second)
        easy_tf_log.tflog('misc/steps', int(step_counter))
        easy_tf_log.tflog('misc/updates', int(update_counter))
        prev_t = cur_t
        prev_steps = cur_steps

        alive = [t.is_alive() for t in worker_threads]
        if not any(alive):
            break


if __name__ == '__main__':
    main()
