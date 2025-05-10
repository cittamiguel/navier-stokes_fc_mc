import subprocess
import argparse
import sys


flags_map ={
    "gcc" : [
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-Ofast",
    ],
    "clang" : [
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-Ofast",
    ]
}

def build(compiler, flags):
    command = f"make clean && make CC={compiler} CFLAGS='{flags}'"
    try:
        proc = subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"{compiler} with flags {flags} run successfully")

    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.stdout}")
        print(f"Command stderr: {e.stderr}")

def run_benchmark(name, mode):
    command = f"make name={name} mode={mode} benchmark"
    try:
        subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"{name} benchmark successful")

    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.stdout}")
        print(f"Command stderr: {e.stderr}")

def run_plot(mode):
    command = f"make name=compilers mode={mode} plot"
    try:
        subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("plot successful")

    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.stdout}")
        print(f"Command stderr: {e.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["headless", "demo"],
        default="headless",
        help= "Mode of operation. Can be 'headless' or 'demo'"
    )
    arg = parser.parse_args()

    for compiler,flags_list in flags_map.items():
        for flags in flags_list:
            build(compiler, flags)

            name = f"{compiler}_{flags.replace('-', '').replace(' ', '_')}"
            run_benchmark(name, arg.mode)
    
    run_plot(arg.mode)


