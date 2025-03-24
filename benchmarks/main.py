import argparse
import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import pandas as pd

DEFAULT_N_VALUES = [256, 500, 864, 1372]
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

    print(f"Running benchmark with {n_values} grid size {num_runs} time/s")
    print(f"Warming up for N = {n_values[0]}...", end=" ", flush=True)
    run_navier_stokes(N=n_values[0], submode=submode) #warmup
    print("ok")

    for n in n_values:
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

    print(f"avg_perfomances:{avg_perfomances} std_devs:{std_devs}")
    save_to_file(name, n_values, avg_perfomances, std_devs)

def get_performance_from_output(output: str) -> float:
    ns_per_cell_pattern = r"^\d+\.?\d+"

    ns_per_cell_match = re.findall(ns_per_cell_pattern, output, re.MULTILINE)

    if not ns_per_cell_match:
        raise ValueError("Missing information in the output")

    ns_per_cell:float = np.mean(
        [float(s) for s in ns_per_cell_match],
        dtype=float
    )
    
    return ns_per_cell

def save_to_file(name: str, n_values: List[int], avg_times: List[float], std_devs: List[float]):
    stats_file = os.path.join("benchmarks/stats", f"{name}.csv")

    if os.path.exists(stats_file):
        df = pd.read_csv(stats_file)
    else:
        df = pd.DataFrame(columns=["N", "avg", "std"])

    for n, avg, std in zip(n_values, avg_times, std_devs):
        if(df.N == n).any():
           df.loc[df.N == n, ["avg", "std"]] = avg, std
        else:
            new_row = pd.DataFrame({"N":[n], "avg":[avg], "std":[std]})
            df = pd.concat([df, new_row], ignore_index=True)
           
    df.to_csv(stats_file, index=False)
            


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
        #plot_benchmark_stats(args.name, args.n_values, args.num_runs, args.submode)
        c = 0

    