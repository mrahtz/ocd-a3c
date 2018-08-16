#!/usr/bin/env python3

"""
Show data (e.g. observations) dumped from the network in debug mode.

To get that data, run with --debug and pipe stderr to a log file, e.g.:
  python3 train.py PongNoFrameskip-v4 --debug 2> log
"""

import argparse
import hashlib
import re

from pylab import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=argparse.FileType('r'))
    args = parser.parse_args()

    for line in args.log_file:
        # We're looking for something like
        #   debug observations:[data]

        if not line.startswith('debug'):
            continue

        tag, data_str = line.split(':')
        data = parse_array(data_str)
        data_type = tag.split(' ')[1]

        if data_type == 'observations':
            show_observations(data)
        elif data_type == 'returns':
            plot_data(data, 'Returns')
        elif data_type == 'actions':
            plot_data(data, 'Actions')


def parse_array(line):
    # Massage tf.Print output into a form that we can eval()
    line = re.sub('([0-9]) ', '\\1, ', line)
    line = re.sub('\]', '], ', line)

    arr = np.array(eval(line)[0])
    sha = hashlib.sha256(str.encode(line)).hexdigest()[:8]
    print("Found array with shape {}; sha256 {}".format(arr.shape, sha))

    return arr


def show_observations(arr):
    obs = arr
    if obs.shape == (1, 80, 80, 4) or obs.shape == (1, 84, 84, 4):
        # A single frame (passed through the network when selecting an action)
        # Stack frames in stack (axis 3) horizontally
        obs = obs[0]
        obs = np.moveaxis(obs, 2, 0)
        obs = np.hstack(obs)
    elif obs.shape[1:] == (80, 80, 4) or obs.shape[1:] == (84, 84, 4):
        # A batch of frames (passed through the network during training)
        # Stack batch items (axis 0) vertically
        obs = np.vstack(obs)
        # Stack frames in stack (axis 3) horizontally
        obs = np.moveaxis(obs, 2, 0)
        obs = np.hstack(obs)
    else:
        print("Unsure how to deal with shape; skipping")
        return
    imshow(obs, cmap='gray')
    show()


def plot_data(data, data_type):
    plot(data)
    ylabel(data_type)
    xlabel("Step")
    tight_layout()
    show()


if __name__ == '__main__':
    main()
