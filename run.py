#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import tensorflow as tf
import time

from worker import Worker
from network import create_network


def worker(i, ckpt_freq, load_ckpt_file):
    """
    Set up a single worker.

    I'm still not 100% about how Distributed TensorFlow works, but as I
    understand it we do "between-graph replication": each worker has a separate
    graph, with the global set of parameters shared between all workers (pinned
    to worker 0).
    """
    dirname = 'summaries/%d_worker%d' % (int(time.time()), i)
    os.makedirs(dirname)
    summary_writer = tf.summary.FileWriter(dirname, flush_secs=1)

    tf.reset_default_graph()
    server = tf.train.Server(cluster, job_name="worker", task_index=i)
    sess = tf.Session(server.target)

    with tf.device("/job:worker/task:0"):
        create_network('global')
    with tf.device("/job:worker/task:%d" % i):
        w = Worker(sess, i, 'PongNoFrameskip-v4', summary_writer)

    if i == 0:
        saver = tf.train.Saver()
        checkpoint_file = os.path.join('checkpoints', 'network.ckpt')

    print("Waiting for cluster cluster connection...")
    sess.run(tf.global_variables_initializer())

    if load_ckpt_file is not None:
        print("Restoring from checkpoint '%s'..." % load_ckpt_file,
              end='', flush=True)
        saver.restore(sess, load_ckpt_file)
        print("done!")

    print("Cluster established!")
    step = 0
    while True:
        print("Step %d" % step)
        done = w.run_step()
        if done:
            w.reset_env()
        step += 1
        if (i == 0) and (step % ckpt_freq == 0):
            print("Saving checkpoint at step %d..." % step, end='', flush=True)
            saver.save(sess, checkpoint_file)
            print("done!")


parser = argparse.ArgumentParser()
parser.add_argument("n_workers", type=int)
parser.add_argument("worker_n", type=int)
parser.add_argument("--port_start", type=int, default=2200)
parser.add_argument("--ckpt_freq", type=int, default=5)
parser.add_argument("--load_ckpt")
args = parser.parse_args()

cluster_dict = {}
workers = []
for i in range(args.n_workers):
    port = args.port_start + i
    workers.append("localhost:%d" % port)
cluster_dict["worker"] = workers
cluster = tf.train.ClusterSpec(cluster_dict)

worker(args.worker_n, args.ckpt_freq, args.load_ckpt)
