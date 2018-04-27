#!/usr/bin/env python3

"""
Want to check exactly what the observations going into your network look like?
Add a line like this to your graph:

    obs = tf.Print(obs, [obs],
                   message='Observations: ',
                   summarize=2147483647)  # max int32

Then run this script on the output of your run to view each observation saved.

In this code:

        x = tf.Print(x, [graph_s],
                     message='Observations: ',
                     summarize=2147483647)  # max int32

        advantage = tf.Print(advantage, [graph_r],
                     message='Returns: ',
                     summarize=2147483647)  # max int32
"""

import argparse
import hashlib
import re

from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('log_file', type=argparse.FileType('r'))
args = parser.parse_args()


def parse_array(line):
    # Massage into a form that we can eval()
    line = re.sub('([0-9]) ', '\\1, ', line)
    line = re.sub('\]', '], ', line)

    array = np.array(eval(line)[0])
    print("Found array with shape", array.shape)
    print("sha256:", hashlib.sha256(str.encode(line)).hexdigest())

    return array


def show_observations(array):
    obs = array
    if obs.shape == (1, 80, 80, 4) or obs.shape == (1, 84, 84, 4):
        obs = obs[0]
        obs = np.moveaxis(obs, 2, 0)
        obs = np.hstack(obs)
    elif obs.shape[1:] == (80, 80, 4) or obs.shape[1:] == (84, 84, 4):
        # Stack axis 0 vertically
        obs = np.vstack(obs)
        # Stack axis 3 horizontally
        obs = np.moveaxis(obs, 2, 0)
        obs = np.hstack(obs)
    else:
        print("Unsure how to deal with shape; skipping")
        return
    imshow(obs, cmap='gray')
    show()


def show_returns(array):
    returns = array
    plot(returns)
    ylabel("Return")
    xlabel("Step")
    show()


for line in args.log_file:
    prefix = 'Observations: '
    if line.startswith(prefix):
        line = line[len(prefix):]
        show_observations(parse_array(line))
    prefix = 'Returns: '
    if line.startswith(prefix):
        line = line[len(prefix):]
        show_returns(parse_array(line))
