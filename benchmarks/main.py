import argparse
import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import pandas as pd

DEFAULT_N_VALUES = [128, 500, 800, 1300]
DEFAULT_NUM_RUNS = [4, 3, 2, 1]

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

def benchmark(name: str, n_values:List[int], num_runs: List[int], submode: str): #num_runs : Int
    avg_perfomances = []
    std_devs = []

    print(f"Running benchmark with {n_values} grid size {num_runs} time/s")
    print(f"Warming up for N = {n_values[0]}...", end=" ", flush=True)

    #run_navier_stokes(N=n_values[0], submode=submode) #warmup

    print("ok")

    for n in range(len(n_values)):
        run_metrics = []
        i = num_runs[n]

        while i != 0:
            print(f"N = {n_values[n]}: {num_runs[n]-i+1}/{num_runs[n]}...", end=" ", flush=True)
            metric = get_performance_from_output(run_navier_stokes(n_values[n], submode))
            run_metrics.append(metric)
            i = i-1
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

def plot_benchmark_stats(name: str, submode: str):
    stats_files = sorted(
        Path("benchmarks/stats").glob("*.csv"), key=os.path.getctime
    )

    plt.figure(figsize=(10, 10))

    for filepath in stats_files:
        print(f"Reading {filepath}...")
        df = pd.read_csv(filepath)
        df.sort_values(by="N")

        n_values = df["N"].to_numpy()
        avg_ns_per_cell = df["avg"].to_numpy()
        std_devs = df["std"].to_numpy()

        plt.errorbar(n_values, avg_ns_per_cell, yerr=std_devs,
                     capsize=5, label=filepath.name)
        
    plt.xticks([256, 500, 800, 1300])
    plt.xlabel("Grid size")
    plt.ylabel("ns per cell")
    plt.legend()
    plt.title(f"{name} stats _ {submode}")

    plt.savefig(f"benchmarks/plots/{name}_{submode}.png", dpi=500)

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
        plot_benchmark_stats(args.name, args.submode)

    