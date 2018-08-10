#!/usr/bin/env python3

"""
Run a trained agent from a checkpoint.
"""

import argparse
import time

import gym
import numpy as np
import tensorflow as tf

from network import make_inference_network
from preprocessing import generic_preprocess


def main():
    args = parse_args()
    env = gym.make(args.env_id)
    env = generic_preprocess(env, max_n_noops=0)
    sess, obs_placeholder, action_probs_op = \
        get_network(args.ckpt_dir, env.action_space.n)
    run_agent(env, sess, obs_placeholder, action_probs_op)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("ckpt_dir")
    args = parser.parse_args()
    return args


def get_network(ckpt_dir, n_actions):
    sess = tf.Session()

    with tf.variable_scope('global'):
        obs_placeholder, _, action_probs_op, _, _ = \
            make_inference_network(n_actions, debug=False)

    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    if not ckpt_file:
        raise Exception("Couldn't find checkpoint in '{}'".format(ckpt_dir))
    print("Loading checkpoint from '{}'".format(ckpt_file))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    return sess, obs_placeholder, action_probs_op


def run_agent(env, sess, obs_placeholder, action_probs_op):
    while True:
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            s = np.moveaxis(obs, 0, -1)
            feed_dict = {obs_placeholder: [s]}
            action_probs = sess.run(action_probs_op, feed_dict)[0]
            action = np.random.choice(env.action_space.n, p=action_probs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(1 / 60.0)
        print("Episode reward:", episode_reward)


if __name__ == '__main__':
    main()
