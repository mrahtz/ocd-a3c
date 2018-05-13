#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import time
from threading import Thread

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


def run_worker(worker, n_steps_to_run, ckpt_timer, steps_per_update):
    updates = 0
    steps = 0
    ckpt_timer.reset()
    while steps < n_steps_to_run:
        start_time = time.time()

        steps_ran = worker.run_update(steps_per_update)
        steps += steps_ran
        updates += 1

        end_time = time.time()
        steps_per_second = steps_ran / (end_time - start_time)

        easy_tf_log.tflog('misc/steps_per_second', steps_per_second)
        easy_tf_log.tflog('misc/steps', steps)
        easy_tf_log.tflog('misc/updates', updates)

        if worker.worker_n == 0 and ckpt_timer.done():
            saver.save(sess, checkpoint_file, steps)
            print("Checkpoint saved to '{}'".format(checkpoint_file))
            ckpt_timer.reset()


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
parser.add_argument("--steps_per_update", type=int, default=5)
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
                       n_ports=(args.n_workers + 1),
                       random_stagger=True)
cluster_dict["parameter_server"] = ["localhost:{}".format(ports[0])]
cluster_dict["worker"] = ["localhost:{}".format(port)
                          for port in ports[1:]]
cluster = tf.train.ClusterSpec(cluster_dict)


def start_parameter_server():
    server = tf.train.Server(cluster, job_name="parameter_server")
    server.join()


def start_worker_process(worker):
    run_worker(worker=worker,
               n_steps_to_run=args.n_steps,
               ckpt_timer=ckpt_timer,
               steps_per_update=args.steps_per_update)


def make_workers(env_id, max_n_noops, debug, n_workers, seed):
    dummy_env = gym.make(env_id)
    create_network('global', n_actions=dummy_env.action_space.n)

    print("Starting {} workers".format(n_workers))
    workers = []
    for worker_n in range(n_workers):
        env = gym.make(env_id)
        seed = seed * n_workers + worker_n
        env.seed(seed)
        if args.debug:
            env = NumberFrames(env)
        env = preprocess_wrapper(env, max_n_noops)
        env = MonitorEnv(env, "Worker {}".format(worker_n))

        worker_log_dir = osp.join(log_dir, "worker_{}".format(worker_n))
        os.makedirs(worker_log_dir)

        w = Worker(sess=sess,
                   env=env,
                   worker_n=worker_n,
                   log_dir=worker_log_dir,
                   max_n_noops=max_n_noops,
                   debug=debug)
        workers.append(w)

    # TODO
    easy_tf_log.set_writer(w.summary_writer.event_writer)

    return workers


utils.set_random_seeds(args.seed)

sess = tf.Session()

workers = make_workers(args.env_id, args.max_n_noops, args.debug,
                       args.n_workers, args.seed)

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

worker_threads = []
for worker_n in range(args.n_workers):
    p = Thread(target=start_worker_process, args=(workers[worker_n],),
               daemon=True)
    p.start()
    worker_threads.append(p)

for p in worker_threads:
    p.join()
