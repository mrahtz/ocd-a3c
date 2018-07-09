#!/usr/bin/env python3

import argparse

import gym

from debug_wrappers import NumberFrames, ConcatFrameStack
from preprocessing import generic_preprocess, pong_preprocess

"""
Play a game with preprocessing applied to check that preprocessing is sane.
"""


def play(env, wrap_fn):
    from gym.utils import play as gym_play
    env = gym.make(env)
    env = NumberFrames(env)
    env = wrap_fn(env, max_n_noops=0)
    env = ConcatFrameStack(env)
    gym_play.play(env, fps=15, zoom=4, callback=callback)


def callback(prev_obs, obs, action, rew, env_done, info):
    print("Step {}: reward {}, done {}".format(callback.step_n,
                                               rew,
                                               env_done))
    callback.step_n += 1
callback.step_n = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env")
    parser.add_argument("--pong_preprocessing", action="store_true")
    args = parser.parse_args()

    if args.pong_preprocessing:
        wrap_fn = pong_preprocess
    else:
        wrap_fn = generic_preprocess
    play(args.env, wrap_fn)


if __name__ == '__main__':
    main()
