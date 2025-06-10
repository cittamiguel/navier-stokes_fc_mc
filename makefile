# Compiler and flags
NVCC       = nvcc
CC         = gcc
CFLAGS     = -O2 -Wall
NVCCFLAGS  = -O2 -Xcompiler -fopenmp

# Source files
C_SOURCES   = $(wildcard *.c)
CU_SOURCES  = $(wildcard *.cu)
OBJECTS     = $(C_SOURCES:.c=.o) $(CU_SOURCES:.cu=.o)

# Output binary
TARGET = main

# Default rule
all: $(TARGET)

# Link all objects into final executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)
