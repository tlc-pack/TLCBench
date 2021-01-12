"""
Print search time for all logs
"""

import argparse
import glob
import json
import os


def get_search_time_autotvm(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        t1 = json.loads(lines[0])['result'][-1]
        t2 = json.loads(lines[-1])['result'][-1]

    return t2 - t1, len(lines)


def get_search_time_autoscheduler(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        t1 = json.loads(lines[0])['r'][-1]
        t2 = json.loads(lines[-1])['r'][-1]

    return t2 - t1, len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir", type=str, default="tmp_logs/", help="Log file directory."
    )
    args = parser.parse_args()

    print("-" * 70)
    print("Time (s) | #Line | Log file")
    print("-" * 70)

    filenames = glob.glob(os.path.join(args.logdir, "autotvm", "*", "*.kernel.log"))
    filenames.sort()
    for filename in filenames:
        time, lines = get_search_time_autotvm(filename)
        print("%8d | %6d | %s" % (time, lines, filename[len(args.logdir):]))

    print("")

    filenames = glob.glob(os.path.join(args.logdir, "autoscheduler", "*", "*.json"))
    filenames.sort()
    for filename in filenames:
        time, lines = get_search_time_autoscheduler(filename)
        print("%8d | %6d | %s" % (time, lines, filename[len(args.logdir):]))

    print("-" * 70)

