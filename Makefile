CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=solver.o wtime.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

benchmark:
	python3 benchmark/main.py $(name) --mode benchmark $(mode) --n-values 256 500 864 1372 2048 --num-runs 3

plot:
	python3 benchmark/main.py $(name) --mode plot $(mode)

clean:
	rm -f $(TARGETS) *.o .depend *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
