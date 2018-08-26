#!/usr/bin/env python3

"""
Plot scores from each of the five main games (Beamrider, Breakout, Pong, Q*bert and Space Invaders)
in a similar format to Figure 4 from the paper.

We split each run into one-hour bins and take the mean score from each bin (such that the score
for e.g. hour 0 is the mean score for the first hour of training).

Error regions are calculated based on the minimum and maximum mean score for each bin for each run.
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from pylab import plot, ylabel, xlabel, title, ylim, xlim, grid, subplot, tight_layout, \
    matplotlib, figure, savefig, fill_between, xticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs_directory')
    args = parser.parse_args()
    runs = [f for f in os.listdir(args.runs_directory)
            if os.path.isdir(os.path.join(args.runs_directory, f)) and '.git' not in f]
    figure(figsize=(24, 4))
    matplotlib.rcParams.update({'font.size': 12})
    for game_n, game in enumerate(['BeamRider', 'Breakout', 'Pong', 'Qbert', 'SpaceInvaders']):
        subplot(1, 5, 1 + game_n)
        plot_game(game, [os.path.join(args.runs_directory, run) for run in runs if game in run])
    tight_layout(pad=0.0, w_pad=1.0)
    savefig('scores.png')


def get_scores_binned_by_hour(run_dir):
    events_filenames = glob.glob(os.path.join(run_dir, 'env_0', 'events.*'))
    assert len(events_filenames) == 1
    events_filename = events_filenames[0]
    times = []
    scores = []
    for event in tf.train.summary_iterator(events_filename):
        for value in event.summary.value:
            if 'episode_reward_sum' not in value.tag:
                continue
            times.append(event.wall_time)
            scores.append(value.simple_value)
    assert sorted(times) == times
    times_from_start_hours = (np.array(times) - times[0]) / 60.0 / 60.0
    bins = np.arange(0.0, 16.0, 1.0)
    # - 1 because otherwise the bins are 1-indexed
    bin_idxs = np.digitize(times_from_start_hours, bins) - 1
    scores_by_bin = []
    for bin_n in range(len(bins)):
        bin_mean = np.mean([scores[i] for i in range(len(scores)) if bin_idxs[i] == bin_n])
        scores_by_bin.append(bin_mean)
    return bins, scores_by_bin


def plot_game(game, run_dirs):
    scores_by_bin_each_seed = []
    for run_dir in run_dirs:
        bins, scores_by_bin = get_scores_binned_by_hour(run_dir)
        scores_by_bin_each_seed.append(scores_by_bin)
    scores_by_bin_each_seed = np.array(scores_by_bin_each_seed)
    scores_by_bin_min = [np.min(scores_by_bin_each_seed[:, n]) for n in range(len(bins))]
    scores_by_bin_avg = [np.mean(scores_by_bin_each_seed[:, n]) for n in range(len(bins))]
    scores_by_bin_max = [np.max(scores_by_bin_each_seed[:, n]) for n in range(len(bins))]
    plot(bins, scores_by_bin_avg)
    fill_between(bins, scores_by_bin_min, scores_by_bin_max, alpha=0.3)
    xlabel("Training time (hours)")
    ylabel("Score")
    xlim([0, 15])
    xticks([0, 2, 4, 6, 8, 10, 12, 14])
    titles = {
        'BeamRider': 'Beamrider',
        'Breakout': 'Breakout',
        'Pong': 'Pong',
        'Qbert': 'Q*bert',
        'SpaceInvaders': 'Space Invaders'
    }
    title(titles[game])
    ylims = {
        'BeamRider': [0, 16000],
        'Breakout': [0, 600],
        'Pong': [-30, 30],
        'Qbert': [0, 12000],
        'SpaceInvaders': [0, 1600]
    }
    ylim(ylims[game])
    grid()


if __name__ == '__main__':
    main()
