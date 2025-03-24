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

def run_navier_stokes(N: int, submode: str) -> str:
    exec = f"./{submode}"

    result = subprocess.run(
        [exec, str(N)],
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout

def benchmark(name: str, n_values:List[int], num_runs: int, submode: str):
    avg_perfomances = []
    std_devs = []

    print(f"Running benchmark with {n_values} particules {num_runs} times")
    for n in n_values:
        print(f"Warming up for N = {n}...", end=" ", flush=True)
        run_navier_stokes(N=n, submode=submode) #warmup
        print("ok")
    
        run_metrics = []
        for i in range(num_runs):
            print(f"N = {n}: {i+1}/{num_runs}...", end=" ", flush=True)
            metric = get_performance_from_output(run_navier_stokes(n, submode))
            run_metrics.append(metric)
            print("ok")

        avg_performance = np.mean(run_metrics)
        std_dev = np.std(run_metrics)

        avg_perfomances.append(avg_performance)
        std_devs.append(std_dev)

    save_to_file(name, n_values, avg_perfomances, std_devs)

def get_performance_from_output(output: str):



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        help= "Name for the benchmark. Used as the filename of the output csv file or plot."
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "plot"],
        default="benchmark-headless",
        help= "Mode of operation. Can be 'benchmark' or 'plot'"
    )
    parser.add_argument(
        "--submode",
        choices=["headless", "demo"],
        default= "headless",
        help="Mode of operation. Can be 'headless' or 'demo'"
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

    if args.mode == "benchmark":
        benchmark(args.name, args.n_values, args.num_runs, args.submode)

    elif args.mode == "plot":
        plot_benchmark_stats(args.name, args.n_values, args.num_runs, args.submode)

    