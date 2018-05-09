#!/usr/bin/env python3
"""
Download all TensorFlow event files from the specified jobs' output files.
"""

import argparse
import os
import os.path as osp
import random
import shutil
import subprocess
import time
from multiprocessing import Pool


def list_files(job_id):
    time.sleep(random.random())
    print("Listing files...")
    cmd = "floyd data listfiles {}/output".format(job_id)
    files = subprocess.check_output(cmd.split()).decode().split('\n')
    event_files = [f for f in files if 'events.out.tfevents' in f]
    job_ids_with_files = [(job_id, f) for f in event_files]
    return job_ids_with_files


def get_file(job_id, job_path, download_dir):
    time.sleep(random.random())
    full_dir = osp.join(download_dir, job_id, osp.dirname(job_path))
    os.makedirs(full_dir, exist_ok=True)
    shutil.copyfile('.floydexpt', osp.join(full_dir, '.floydexpt'))

    print("Downloading {}...".format(job_path))
    cmd = "floyd data getfile {}/output {}".format(job_id, job_path)
    # We need to download directly into the target directory in case
    # there are multiple files with the same name (which would overwrite
    # each other if downloaded into the same directory).
    subprocess.call(cmd.split(), cwd=full_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("download_dir")
    parser.add_argument("job_ids", nargs='*')
    parser.add_argument("--n_parallel", default=8)
    args = parser.parse_args()

    with Pool(processes=args.n_parallel) as pool:
        job_ids_with_files = pool.map(list_files, args.job_ids)
        job_ids_with_files = [
            item for sublist in job_ids_with_files for item in sublist
        ]
        worker_args = [(job_id, job_path, args.download_dir)
                       for job_id, job_path in job_ids_with_files]
        pool.starmap(get_file, worker_args)
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
