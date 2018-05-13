#!/usr/bin/env python3

import os
import os.path as osp
import time
from threading import Thread

import easy_tf_log
import gym
import tensorflow as tf

import utils
from debug_wrappers import NumberFrames, MonitorEnv
from network import create_network
from params import parse_args
from worker import Worker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def run_worker(worker, n_steps_to_run, steps_per_update, step_counter,
               update_counter):
    while int(step_counter) < n_steps_to_run:
        steps_ran = worker.run_update(steps_per_update)
        step_counter.increment(steps_ran)
        update_counter.increment(1)


def make_workers(sess, env_id, preprocess_wrapper, max_n_noops, debug,
                 n_workers, seed, log_dir):
    dummy_env = gym.make(env_id)
    create_network('global', n_actions=dummy_env.action_space.n)

    print("Starting {} workers".format(n_workers))
    workers = []
    for worker_n in range(n_workers):
        env = gym.make(env_id)
        seed = seed * n_workers + worker_n
        env.seed(seed)
        if debug:
            env = NumberFrames(env)
        env = preprocess_wrapper(env, max_n_noops)
        env = MonitorEnv(env, "worker_{}".format(worker_n))

        worker_log_dir = osp.join(log_dir, "worker_{}".format(worker_n))
        os.makedirs(worker_log_dir)

        w = Worker(sess=sess,
                   env=env,
                   worker_n=worker_n,
                   log_dir=worker_log_dir,
                   max_n_noops=max_n_noops,
                   debug=debug)
        workers.append(w)

    return workers


def start_workers(args, step_counter, update_counter, workers):
    worker_threads = []
    for worker_n, worker in enumerate(workers):
        p_args = (worker,
                  args.n_steps,
                  args.steps_per_update,
                  step_counter,
                  update_counter)
        p = Thread(target=run_worker, args=p_args)
        p.start()
        worker_threads.append(p)
    return worker_threads


def main():
    args, log_dir, preprocess_wrapper, ckpt_timer = parse_args()
    easy_tf_log.set_dir(log_dir)

    sess = tf.Session()
    utils.set_random_seeds(args.seed)
    workers = make_workers(sess, args.env_id,
                           preprocess_wrapper, args.max_n_noops, args.debug,
                           args.n_workers, args.seed, log_dir)

    # Why save_relative_paths=True?
    # So that the plain-text 'checkpoint' file written uses relative paths,
    # which seems to be needed in order to avoid confusing saver.restore()
    # when restoring from FloydHub runs.
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    checkpoint_dir = osp.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir)
    checkpoint_file = osp.join(checkpoint_dir, 'network.ckpt')

    if args.load_ckpt:
        print("Restoring from checkpoint '%s'..." % args.load_ckpt,
              end='', flush=True)
        saver.restore(sess, args.load_ckpt)
        print("done!")
    else:
        sess.run(tf.global_variables_initializer())

    step_counter = utils.ThreadSafeCounter()
    update_counter = utils.ThreadSafeCounter()

    worker_threads = start_workers(args, step_counter, update_counter, workers)

    ckpt_timer.reset()
    prev_t = time.time()
    prev_steps = int(step_counter)
    while True:
        time.sleep(1.0)

        cur_t = time.time()
        cur_steps = int(step_counter)
        steps_per_second = (cur_steps - prev_steps) / (cur_t - prev_t)
        easy_tf_log.tflog('misc/steps_per_second', steps_per_second)
        easy_tf_log.tflog('misc/steps', int(step_counter))
        easy_tf_log.tflog('misc/updates', int(update_counter))

        if ckpt_timer.done():
            saver.save(sess, checkpoint_file, int(step_counter))
            print("Checkpoint saved to '{}'".format(checkpoint_file))
            ckpt_timer.reset()

        alive = [t.is_alive() for t in worker_threads]
        if not any(alive):
            break


if __name__ == '__main__':
    main()
