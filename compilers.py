import subprocess
import argparse

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
    ],
    "gcc-12" : [
        "-Ofast",
    ]
} #TODO: add the intel compiler and corresponding flags

def build(compiler, flags):
    command = f"make clean && make CC={compiler} CFLAGS='{flags}'"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print(f"{compiler} with flags {flags} run successfully")

    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.output}")

def run_benchmark(name, mode):
    command = f"make name={name} mode={mode} benchmark"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print(f"{name} benchmark successful")

    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.output}")

def run_plot(mode):
    command = f"make name=compilers mode={mode} plot"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print("plot successful")

    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"failed with return code {e.returncode}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.output}")
#TODO: make changes on make file to be able to run run_benchmark(name) & run_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["headless", "demo"],
        default="headless",
        help= "Mode of operation. Can be 'headless'or 'demo'"
    )
    arg = parser.parse_args()

    for compiler,flags_list in flags_map.items():
        for flags in flags_list:
            build(compiler, flags)

            name = f"{compiler}_{flags.replace('-', '').replace(' ', '_')}"
            run_benchmark(name, arg.mode)
    
    run_plot(arg.mode)


