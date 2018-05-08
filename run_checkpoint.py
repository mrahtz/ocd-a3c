#!/usr/bin/env python3

"""
Run a trained agent.
"""

import argparse
import time

import gym
import numpy as np
import tensorflow as tf

from network import create_network
from preprocessing import generic_preprocess


def main():
    args = parse_args()
    env = gym.make(args.env_id)
    env = generic_preprocess(env, max_n_noops=0)
    sess, network = get_network(args.ckpt_dir, env.action_space.n)
    run_agent(env, sess, network)


def run_agent(env, sess, network):
    while True:
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            s = np.moveaxis(obs, 0, -1)
            feed_dict = {network.s: [s]}
            action_probs = sess.run(network.a_softmax, feed_dict)[0]
            action = np.random.choice(env.action_space.n, p=action_probs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(1/60.0)
        print("Episode reward:", episode_reward)


def get_network(ckpt_dir, n_actions):
    sess = tf.Session()
    network = create_network(n_actions=n_actions, scope='worker_0')
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)
    return sess, network


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("ckpt_dir")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
