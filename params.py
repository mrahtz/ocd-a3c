import argparse
import os
import time
from os import path as osp

import preprocessing
from utils import get_git_rev, Timer


def parse_args():
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
    if args.preprocessing == 'generic':
        preprocess_wrapper = preprocessing.generic_preprocess
    elif args.preprocessing == 'pong':
        preprocess_wrapper = preprocessing.pong_preprocess
    ckpt_timer = Timer(duration_seconds=args.ckpt_interval_seconds)

    return args, log_dir, preprocess_wrapper, ckpt_timer


DISCOUNT_FACTOR = 0.99