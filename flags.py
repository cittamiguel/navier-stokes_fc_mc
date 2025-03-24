import subprocess
import argparse

flags_list = [
    "-O0",
    "-O1",
    "-O2",
    "-O3",
    "-Ofast",
    "-O3 -march=native",
    "-Ofast -march=native",
    "-Ofast -march=native -funsafe-math-optimizations",
    "-Ofast -march=native -ffinite-math-only",
    "-Ofast -march=native -funsafe-math-optimizations -ffinite-math-only",
    "-Ofast -march=native -funsafe-math-optimizations -ffinite-math-only -funroll-loops",
]


def build(flags):
    command = f"make clean && make CC=gcc CFLAGS='{flags}'"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print(f"compile successful w/ {flags}")
        
    except subprocess.CalledProcessError as e:
        print(f"make clean && make CC=gcc CFLAGS='{flags}'")
        print(f"failed w/ return code {e.returncode}")
        print(f"Command output: {e.stdout}")
        print(f"Command stderr: {e.stderr}")

def run_benchmark(name, mode):
    command = f"make name={name} mode={mode} benchmark"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print(f"{name} benchmark-{mode} successful")
        
    except subprocess.CalledProcessError as e:
        print(f"{command} with mode {mode}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.output}")

def run_plot(mode):
    command = f"make name=flags mode={mode} plot"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print(f"flags plot-{mode} successful")
        
    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.output}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["headless", "demo"],
        default="headless",
        help= "Mode of operation. Can be 'headless'or 'demo'"
    )
    arg = parser.parse_args()

    for flags in flags_list:
        build(flags)
        run_benchmark(f"gcc_{flags.replace('-', '').replace(' ', '_')}", arg.mode)

    run_plot(arg.mode)
