#
# Simulation of rainwater flooding
# 
#

# Compilers
CC=gcc
CUDACC=nvcc

# Flags for optimization and libs
FLAGS=-O3 -Wall
CUFLAGS=-O3 
LIBS=-lm
CULIBS=-lm rng.c

# Targets to build
OBJS=flood_seq flood_cuda

# Rules. By default show help
help:
	@echo
	@echo "Simulation of rainwater flooding"
	@echo
	@echo "make flood_seq	Build only the sequential version"
	@echo "make flood_cuda	Build only the CUDA version"
	@echo
	@echo "make all		Build all versions (Sequential & CUDA)"
	@echo "make debug		Build sequential version with demo output for small surfaces"
	@echo "make animation		Build the sequential version to produce the animation data"
	@echo "make clean		Remove targets"
	@echo "make test_seq		Build and run the sequential version with a simple input sequence (not suitable for DAS5)"
	@echo "make test_seq_remote	Build and run the sequential version with a simple input sequence on a compute node"
	@echo "make test_seq_sm		Build and run the seq version with input small_mountains on a compute node and save output in res_seq_sm.out"
	@echo "make test_seq_lm		Build and run the seq version with input large_mountains on a compute node save output in res_seq_lm.out"
	@echo "make test_seq_mld	Build and run the seq version with input medium_lower_dam on a compute node save output in res_seq_mld.out"
	@echo "make test_seq_mhd	Build and run the seq version with input medium_higher_dam on a compute node save output in res_seq_mhd.out"
	@echo "make test_cuda		Build and run the CUDA version with a simple input sequence on a compute node"
	@echo "make test_cuda_sm	Build and run the CUDA version with input small_mountains on a compute node and save output in res_cuda_sm.out"
	@echo "make test_cuda_lm	Build and run the CUDA version with input large_mountains on a compute node save output in res_cuda_lm.out"
	@echo "make test_cuda_mld	Build and run the CUDA version with input medium_lower_dam on a compute node save output in res_cuda_mld.out"
	@echo "make test_cuda_mhd	Build and run the CUDA version with input medium_higher_dam on a compute node save output in res_cuda_mhd.out"
	@echo

all: $(OBJS)

flood.o: flood.c
	$(CC) $(FLAGS) $(DEBUG) -c $< -o $@

flood_seq.o: flood_seq.c
	$(CC) $(FLAGS) $(DEBUG) -c $< -o $@

flood_seq: flood.o flood_seq.o
	$(CC) $(DEBUG) $^ $(LIBS) -o $@

flood_cuda.o: flood_cuda.cu
	$(CUDACC) $(CUFLAGS) $(DEBUG) -c $< -o $@

flood_cuda: flood.o flood_cuda.o
	$(CUDACC) $(DEBUG) $^ $(CULIBS) -o $@

# Remove the target files
clean:
	rm -rf $(OBJS) *.o

# Compile in debug mode (currently sequential version only)
debug:
	make FLAGS="$(FLAGS) -DDEBUG -g" flood_seq

# Compile to generate animation (currently sequential version only)
animation:
	make FLAGS="$(FLAGS) -DDEBUG -DANIMATION -g" flood_seq

animation_cuda:
	make CUFLAGS="$(CUFLAGS) -DDEBUG -DANIMATION -g" FLAGS="$(FLAGS) -DDEBUG -DANIMATION -g" flood_cuda

profile:
	make CUFLAGS="$(CUFLAGS) -DPROFILE" flood_cuda

test_seq: flood_seq
	./flood_seq $(cat test_files/debug.in)

test_seq_remote: flood_seq
	prun -t 15:00 -np 1 -native '-C gpunode' ./flood_seq $$(cat test_files/debug.in)

test_seq_sm: flood_seq
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_seq $$(cat test_files/small_mountains.in) > res_seq_sm.out

test_seq_lm: flood_seq
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_seq $$(cat test_files/large_mountains.in) > res_seq_lm.out

test_seq_mld: flood_seq
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_seq $$(cat test_files/medium_lower_dam.in) > res_seq_mld.out

test_seq_mhd: flood_seq
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_seq $$(cat test_files/medium_higher_dam.in) > res_seq_mhd.out

test_cuda: flood_cuda
	prun -t 15:00 -np 1 -native '-C gpunode' ./flood_cuda $$(cat test_files/debug.in)

test_cuda_sm: flood_cuda
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_cuda $$(cat test_files/small_mountains.in) > res_cuda_sm.out

test_cuda_lm: flood_cuda
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_cuda $$(cat test_files/large_mountains.in) > res_cuda_lm.out

test_cuda_mld: flood_cuda
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_cuda $$(cat test_files/medium_lower_dam.in) > res_cuda_mld.out

test_cuda_mhd: flood_cuda
	prun -t 15:00 -np 1 -native '-C gpunode,TitanX' ./flood_cuda $$(cat test_files/medium_higher_dam.in) > res_cuda_mhd.out
