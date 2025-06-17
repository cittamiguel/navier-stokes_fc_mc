CC         =gcc
CFLAGS     =-std=c11 -Wall -Wextra -Wno-unused-parameter
NVCC       = nvcc
NVCCFLAGS  = -O2 -Xcompiler -fopenmp -arch=compute_61 -code=sm_61

LDFLAGS=

TARGETS         = demo headless
SOURCES         = $(wildcard *.c)
CU_SOURCES      = $(wildcard *.cu)
COMMON_OBJECTS  = wtime.o
CUDA_OBJECT     = solver.o

SOLVER_CFLAGS=-march=native -std=c99 -Werror -Wextra -Rpass=loop-vectorize -ftree-vectorize -ffast-math -funsafe-math-optimizations -O3

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS) $(CUDA_OBJECT)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS) $(CUDA_OBJECT)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_OBJECT): solver.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

benchmark:
	python3 benchmarks/main.py $(name) --mode benchmark --submode $(mode)

plot:
	python3 benchmarks/main.py $(name) --mode plot --submode $(mode)

clean:
	rm -f $(TARGETS) *.o .depend *~

depend:
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
