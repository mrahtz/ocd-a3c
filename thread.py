#!/usr/bin/env python

import argparse
import os
import tensorflow as tf
import time

from worker import Worker
from network import create_network

def worker(i):
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
    server = tf.train.Server(
        cluster, job_name="worker", task_index=i)
    sess = tf.Session(server.target)

    with tf.device("/job:worker/task:0"):
        global_network = create_network('global')
    with tf.device("/job:worker/task:%d" % i):
        w = Worker(sess, i, 'PongNoFrameskip-v4', summary_writer)

    print("Waiting for cluster cluster connection...")
    sess.run(tf.global_variables_initializer())
    print("Cluster established!")
    while True:
        done = w.run_step()
        if done:
            w.reset_env()

parser = argparse.ArgumentParser()
parser.add_argument("n_workers", type=int)
parser.add_argument("worker_n", type=int)
parser.add_argument("--port_start", type=int, default=2200)
args = parser.parse_args()

cluster_dict = {}
workers = []
for i in range(args.n_workers):
    port = args.port_start + i
    workers.append("localhost:%d" % port)
cluster_dict["worker"] = workers
cluster = tf.train.ClusterSpec(cluster_dict)

worker(args.worker_n)
