CC=clang
CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=wtime.o
SOLVER_OBJECT=solver.o
SOLVER_CFLAGS=-march=native -std=c99 -Werror -Wextra -Rpass=loop-vectorize -ftree-vectorize -ffast-math -funsafe-math-optimizations -O3

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS) $(SOLVER_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(SOLVER_OBJECT):
	$(CC) -c $(SOLVER_CFLAGS) solver.c -o solver.o

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
