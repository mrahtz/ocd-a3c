#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import time
from multiprocessing import Process

import tensorflow as tf

from network import create_network
from utils import get_port_range, profile_memory, get_git_rev
from worker import Worker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages



def run_worker(env_id, worker_n, n_steps, ckpt_freq, load_ckpt_file, render,
               log_dir):
    dirname = osp.join(log_dir, "worker_{}".format(worker_n))
    os.makedirs(dirname)
    summary_writer = tf.summary.FileWriter(dirname, flush_secs=1)

    server = tf.train.Server(cluster, job_name="worker", task_index=worker_n)
    sess = tf.Session(server.target)

    with tf.device("/job:worker/task:0"):
        create_network('global')
    with tf.device("/job:worker/task:%d" % worker_n):
        w = Worker(sess, worker_n, env_id, summary_writer)
        if render:
            w.render = True

    if worker_n == 0:
        saver = tf.train.Saver()
        checkpoint_file = os.path.join('checkpoints', 'network.ckpt')

    print("Waiting for cluster connection...")
    sess.run(tf.global_variables_initializer())

    if load_ckpt_file is not None:
        print("Restoring from checkpoint '%s'..." % load_ckpt_file,
              end='', flush=True)
        saver.restore(sess, load_ckpt_file)
        print("done!")

    print("Cluster established!")
    step = 0
    while step < n_steps:
        print("Step %d" % step)
        done = w.run_step()
        if done:
            w.reset_env()
        step += 1
        if (worker_n == 0) and (step % ckpt_freq == 0):
            print("Saving checkpoint at step %d..." % step, end='', flush=True)
            saver.save(sess, checkpoint_file)
            print("done!")


parser = argparse.ArgumentParser()
parser.add_argument("env_id")
parser.add_argument("--n_steps", type=int, default=10)
parser.add_argument("--n_workers", type=int, default=16)
parser.add_argument("--ckpt_freq", type=int, default=5)
parser.add_argument("--load_ckpt")
parser.add_argument("--render", action='store_true')
seconds_since_epoch = str(int(time.time()))
parser.add_argument('--run_name', default=seconds_since_epoch)
args = parser.parse_args()

git_rev = get_git_rev()
run_name = args.run_name + '_' + git_rev
log_dir = osp.join('runs', run_name)
if osp.exists(log_dir):
    raise Exception("Log directory '%s' already exists" % log_dir)
os.makedirs(log_dir)

if "MovingDot" in args.env_id:
    import gym_moving_dot
    gym_moving_dot  # TODO prevent PyCharm from removing the import

cluster_dict = {}
ports = get_port_range(start_port=2200, n_ports=args.n_workers)
cluster_dict["worker"] = ["localhost:{}".format(port)
                          for port in ports]
cluster = tf.train.ClusterSpec(cluster_dict)


def start_worker_process(worker_n):
    print("Starting worker", worker_n)
    run_worker(args.env_id,
               worker_n,
               args.n_steps,
               args.ckpt_freq,
               args.load_ckpt,
               args.render,
               log_dir)


worker_processes = []
memory_profiler_processes = []
for worker_n in range(args.n_workers):
    p = Process(target=start_worker_process, args=(worker_n,), daemon=True)
    p.start()
    worker_processes.append(p)

    # TODO: remove
    mem_log = osp.join(log_dir, "worker_{}_memory.log".format(worker_n))
    mp = profile_memory(mem_log, p.pid)
    memory_profiler_processes.append(mp)


for p in worker_processes:
    print("Joining", p)
    p.join()
print("Joined")
for p in memory_profiler_processes:
    print("Terminating mp", p)
    p.terminate()
print("Terminated")
