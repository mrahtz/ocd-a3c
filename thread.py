#!/usr/bin/env python

import argparse
import os
import tensorflow as tf
import time

from worker import Worker
from network import create_network

def ps():
    tf.reset_default_graph()
    server = tf.train.Server(cluster, job_name="ps")
    sess = tf.Session(server.target)
    with tf.device("/job:ps/task:0"):
        global_network = create_network('global')
    print("Running initialiser...")
    sess.run(tf.global_variables_initializer())
    print("Setup done!")
    server.join()

def worker(i):
    dirname = 'summaries/%d_worker%d' % (int(time.time()), i)
    os.makedirs(dirname)
    summary_writer = tf.summary.FileWriter(dirname, flush_secs=1)

    tf.reset_default_graph()
    server = tf.train.Server(
        cluster, job_name="worker", task_index=i)
    sess = tf.Session(server.target)
    with tf.device("/job:ps/task:0"):
        global_network = create_network('global')
    with tf.device("/job:worker/task:%d" % i):
        w = Worker(sess, i, 'PongNoFrameskip-v4', summary_writer)
    print("Running initialiser...")
    sess.run(tf.global_variables_initializer())
    print("Setup done!")
    while True:
        done = w.run_step()
        if done:
            w.reset_env()

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("total", type=int)
parser.add_argument("--n", type=int)
args = parser.parse_args()

cluster_dict = {}
cluster_dict["ps"] = ["localhost:2300"]
workers = []
for i in range(args.total):
    workers.append("localhost:22%02d" % (i))
cluster_dict["worker"] = workers
print(cluster_dict)
exit()
cluster = tf.train.ClusterSpec(cluster_dict)

if args.mode == "ps":
    ps()
elif args.mode == "worker":
    worker(args.n)
