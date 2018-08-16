#!/usr/bin/env python
"""
Plot process memory usage graphs recorded by utils.profile_memory.
"""

import argparse

from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('mem_log', nargs='*')
args = parser.parse_args()

for i, log in enumerate(args.mem_log):
    with open(log) as f:
        lines = f.read().rstrip().split('\n')
    mems = [float(l.split()[1]) for l in lines]
    times = [float(l.split()[2]) for l in lines]
    rtimes = [t - times[0] for t in times]
    subplot(len(args.mem_log), 1, i + 1)
    title(log)
    plot(rtimes, mems)

tight_layout()
show()
