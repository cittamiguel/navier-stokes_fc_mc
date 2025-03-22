import argparse
import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import pandas as pd

DEFAULT_N_VALUES = [256]
DEFAULT_NUM_RUNS = 2

# add to argparse demo, headless, plot

def run_navier_stokes_headless(N: int) -> str:
    result = subprocess.run(
        ['./headless']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        help= "Name for the benchmark. Used as the filename of the output csv file or plot."
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark-headless", "plot-headless", "benchmark-demo", "plot-demo"],
        default="benchmark-headless",
        help= "Mode of operation. Can be 'benchmark-headless', 'plot-headless', 'benchmark-demo', 'plot-demo'"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        metavar="NUM_RUNS",
        default=DEFAULT_NUM_RUNS,
        help= f"Number of times to run the simulation. Default is {DEFAULT_NUM_RUNS}",
    )
    parser.add_argument(
        "--n-values",
        nargs='+',
        type=int,
        metavar="N",
        default=DEFAULT_N_VALUES,
        help= f"List of N values to use. Default is {DEFAULT_N_VALUES}."
    )

    args = parser.parse_args()

    if args.mode == "benchmark-headless":
        benchmark_headless(args.name, args.n_values, args.num_runs)
    elif args.mode == "benchmark-demo":
        benchmark_demo(args.name, args.n_values, args.num_runs)
    elif args.mode == "plot-headless":
        plot_headless_stats(args.name, args.n_values, args.num_runs)
    elif args.mode == "plot-demo":
        plot_demo_stats(args.name, args.n_values, args.num_runs)

    