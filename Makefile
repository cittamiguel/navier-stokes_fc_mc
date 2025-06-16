CC         =gcc
CFLAGS     =-std=c11 -Wall -Wextra -Wno-unused-parameter
NVCC       = nvcc
NVCCFLAGS  = -O2 -Xcompiler -fopenmp

LDFLAGS=

TARGETS         = demo headless
SOURCES         = $(wildcard *.c)
CU_SOURCES      = $(wildcard *.cu)
COMMON_OBJECTS  = wtime.o
SOLVER_OBJECT   = solver.o
CUDA_OBJECT     = lin_solve.o

SOLVER_CFLAGS=-march=native -std=c99 -Werror -Wextra -Rpass=loop-vectorize -ftree-vectorize -ffast-math -funsafe-math-optimizations -O3

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS) $(SOLVER_OBJECT) $(CUDA_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS) $(SOLVER_OBJECT) $(CUDA_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(SOLVER_OBJECT): solver.c
	$(CC) -c $(SOLVER_CFLAGS) $< -o $@

$(CUDA_OBJECT): lin_solve.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

benchmark:
	python3 benchmarks/main.py $(name) --mode benchmark --submode $(mode)

plot:
	python3 benchmarks/main.py $(name) --mode plot --submode $(mode)

clean:
	rm -f $(TARGETS) *.o .depend *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
