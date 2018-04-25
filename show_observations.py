#!/usr/bin/env python3

"""
Want to check exactly what the observations going into your network look like?
Add a line like this to your graph:

    obs = tf.Print(obs, [obs],
                   message='Observations: ',
                   summarize=2147483647)  # max int32

Then run this script on the output of your run to view each observation saved.
"""

import argparse
import re

from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('log_file', type=argparse.FileType('r'))
args = parser.parse_args()

for line in args.log_file:
    prefix = 'Observations: '
    if not line.startswith(prefix):
        continue
    line = line[len(prefix):]

    # Massage into a form that we can eval()
    line = re.sub('([0-9]) ', '\\1, ', line)
    line = re.sub('\]', '], ', line)

    obs = np.array(eval(line)[0])
    print("Found observation with shape", obs.shape)

    if obs.shape == (1, 84, 84, 4):
        obs = obs[0]
        obs = np.moveaxis(obs, 2, 0)
        obs = np.hstack(obs)
    elif obs.shape == (5, 80, 80, 4) or obs.shape == (5, 84, 84, 4):
        # Stack axis 0 vertically
        obs = np.vstack(obs)
        # Stack axis 3 horizontally
        obs = np.moveaxis(obs, 2, 0)
        obs = np.hstack(obs)
    else:
        print("Unsure how to deal with shape; skipping")
        continue

    imshow(obs, cmap='gray')
    show()
