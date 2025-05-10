CC=cc
CFLAGS=-std=c99 -Wall -Wextra -Wno-unused-parameter
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o
SOLVER_OBJECT=solver.o
SOLVER_CFLAGS=-march=native -mavx2 -mfma -fopt-info -ftree-vectorize -ffast-math -funsafe-math-optimizations -O3

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS) $(SOLVER_OBJECT)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(SOLVER_OBJECT):
	$(CC) -c $(SOLVER_CFLAGS) -g solver.c -o solver.o

clean:
	rm -f $(TARGETS) *.o .depend *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
