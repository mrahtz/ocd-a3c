#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import time
from multiprocessing import Process

import easy_tf_log
import gym
import tensorflow as tf

import preprocessing
import utils
from debug_wrappers import NumberFrames, MonitorEnv
from network import create_network
from utils import get_port_range, MemoryProfiler, get_git_rev, Timer
from worker import Worker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def run_worker(env_id, preprocess_wrapper, seed, worker_n, n_steps_to_run,
               ckpt_timer, load_ckpt_file, render, log_dir, max_n_noops,
               debug, steps_per_update):
    utils.set_random_seeds(seed)

    env = gym.make(env_id)
    env.seed(seed)
    if debug:
        env = NumberFrames(env)
    env = preprocess_wrapper(env, max_n_noops)
    env = MonitorEnv(env, "Worker {}".format(worker_n))

    mem_log = osp.join(log_dir, "worker_{}_memory.log".format(worker_n))
    memory_profiler = MemoryProfiler(pid=-1, log_path=mem_log)
    memory_profiler.start()

    worker_log_dir = osp.join(log_dir, "worker_{}".format(worker_n))
    os.makedirs(worker_log_dir)

    server = tf.train.Server(cluster, job_name="worker", task_index=worker_n)
    sess = tf.Session(server.target)

    with tf.device("/job:worker/task:0"):
        create_network(scope='global', n_actions=env.action_space.n)
    with tf.device("/job:worker/task:%d" % worker_n):
        w = Worker(sess=sess,
                   env=env,
                   worker_n=worker_n,
                   log_dir=worker_log_dir,
                   max_n_noops=max_n_noops,
                   debug=debug)
        init_op = tf.global_variables_initializer()
        if render:
            w.render = True

    easy_tf_log.set_writer(w.summary_writer.event_writer)

    # Worker 0 initialises the global network as well as the per-worker networks
    # Other workers only initialise their own per-worker networks
    sess.run(init_op)

    if worker_n == 0:
        saver = tf.train.Saver()
        checkpoint_dir = osp.join(log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir)
        checkpoint_file = osp.join(checkpoint_dir, 'network.ckpt')

    if load_ckpt_file is not None:
        print("Restoring from checkpoint '%s'..." % load_ckpt_file,
              end='', flush=True)
        saver.restore(sess, load_ckpt_file)
        print("done!")

    updates = 0
    steps = 0
    ckpt_timer.reset()
    while steps < n_steps_to_run:
        start_time = time.time()

        steps_ran = w.run_update(steps_per_update)
        steps += steps_ran
        updates += 1

        end_time = time.time()
        steps_per_second = steps_ran / (end_time - start_time)

        easy_tf_log.tflog('misc/steps_per_second', steps_per_second)
        easy_tf_log.tflog('misc/steps', steps)
        easy_tf_log.tflog('misc/updates', updates)

        if worker_n == 0 and ckpt_timer.done():
            saver.save(sess, checkpoint_file)
            print("Checkpoint saved to '{}'".format(checkpoint_file))
            ckpt_timer.reset()

    memory_profiler.stop()


parser = argparse.ArgumentParser()
parser.add_argument("env_id")
parser.add_argument("--n_steps", type=int, default=10000000)
parser.add_argument("--n_workers", type=int, default=1)
parser.add_argument("--ckpt_interval_seconds", type=int, default=60)
parser.add_argument("--load_ckpt")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--render", action='store_true')
parser.add_argument("--max_n_noops", type=int, default=30)
parser.add_argument("--debug", action='store_true')
parser.add_argument("--steps_per_update", type=int, default=100000)
parser.add_argument("--preprocessing",
                    choices=['generic', 'pong'],
                    default='pong')
group = parser.add_mutually_exclusive_group()
group.add_argument('--log_dir')
seconds_since_epoch = str(int(time.time()))
group.add_argument('--run_name',
                   default='test-run_{}'.format(seconds_since_epoch))
args = parser.parse_args()

if args.log_dir:
    log_dir = args.log_dir
else:
    git_rev = get_git_rev()
    run_name = args.run_name + '_' + git_rev
    log_dir = osp.join('runs', run_name)
    if osp.exists(log_dir):
        raise Exception("Log directory '%s' already exists" % log_dir)
os.makedirs(log_dir, exist_ok=True)

if "MovingDot" in args.env_id:
    import gym_moving_dot

    gym_moving_dot  # TODO prevent PyCharm from removing the import

if args.preprocessing == 'generic':
    preprocess_wrapper = preprocessing.generic_preprocess
elif args.preprocessing == 'pong':
    preprocess_wrapper = preprocessing.pong_preprocess

ckpt_timer = Timer(duration_seconds=args.ckpt_interval_seconds)

cluster_dict = {}
ports = get_port_range(start_port=2200,
                       n_ports=args.n_workers,
                       random_stagger=True)
cluster_dict["worker"] = ["localhost:{}".format(port)
                          for port in ports]
cluster = tf.train.ClusterSpec(cluster_dict)


def start_worker_process(worker_n, seed):
    run_worker(env_id=args.env_id,
               preprocess_wrapper=preprocess_wrapper,
               seed=seed,
               worker_n=worker_n,
               n_steps_to_run=args.n_steps,
               ckpt_timer=ckpt_timer,
               load_ckpt_file=args.load_ckpt,
               render=args.render,
               log_dir=log_dir,
               max_n_noops=args.max_n_noops,
               debug=args.debug,
               steps_per_update=args.steps_per_update)


worker_processes = []
memory_profiler_processes = []
print("Starting {} workers".format(args.n_workers))
for worker_n in range(args.n_workers):
    seed = args.seed * args.n_workers + worker_n
    p = Process(target=start_worker_process, args=(worker_n, seed), daemon=True)
    p.start()
    worker_processes.append(p)

for p in worker_processes:
    p.join()
