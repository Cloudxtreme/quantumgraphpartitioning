# Makefile for HPCSEII Project

#Compiler Configuration.
include config.mk

GRAPHSIZE = 10
# RNDSEED >= 0 for random graph
# RNDSEED = -1 for two subgraphs
# RNDSEED = -2 for circular graph
RNDSEED = 120

.PHONY: run, clean, viz, sync_euler, sync_daint, send_to_euler, send_to_daint

FLAGS = -std=c++11 -Wall -O3 -fopenmp -funroll-loops -fprefetch-loop-arrays -mavx -march=native 
FLAGS_VECOFF = $(FLAGS) -D VECTORIZE=0
FLAGS_VECON = $(FLAGS) -D VECTORIZE=1
FLAGS_VECGCC = $(FLAGS) -D VECTORIZE=-1 -ftree-vectorize -ftree-vectorizer-verbose=1 # -funsafe-math-optimizations 
#-lstdc++
LIBS = #-lboost_system -lboost_filesystem

RUNCMD = mpirun --mca orte_base_help_aggregate 0 -np 2

GRAPHSRC_FILES := $(wildcard *.gv)
GRAPHIMG_FILES := $(GRAPHSRC_FILES:.gv=.png)







all: graphsolver_vecoff.exe graphsolver_vecgcc.exe graphsolver_vecon.exe

graphsolver_vecoff.exe: isingchain_local_vecoff.o qmgraph.o graphsolver_vecoff.o
	$(MPICOMPILER) $(FLAGS_VECOFF) isingchain_local_vecoff.o qmgraph.o graphsolver_vecoff.o -o graphsolver_vecoff.exe $(LIBS)

graphsolver_vecgcc.exe: isingchain_local_vecgcc.o qmgraph.o graphsolver_vecgcc.o
	$(MPICOMPILER) $(FLAGS_VECGCC) isingchain_local_vecgcc.o qmgraph.o graphsolver_vecgcc.o -o graphsolver_vecgcc.exe $(LIBS)

graphsolver_vecon.exe: isingchain_local_vecon.o qmgraph.o graphsolver_vecon.o
	$(MPICOMPILER) $(FLAGS_VECON) isingchain_local_vecon.o qmgraph.o graphsolver_vecon.o -o graphsolver_vecon.exe $(LIBS)


graphsolver_vecoff.o: isingchain_local.hpp qmgraph.hpp graphsolver.cpp
	$(MPICOMPILER) $(FLAGS_VECOFF) -c graphsolver.cpp -o graphsolver_vecoff.o $(LIBS)

graphsolver_vecgcc.o: isingchain_local.hpp qmgraph.hpp graphsolver.cpp
	$(MPICOMPILER) $(FLAGS_VECGCC) -c graphsolver.cpp -o graphsolver_vecgcc.o $(LIBS)

graphsolver_vecon.o: isingchain_local.hpp qmgraph.hpp graphsolver.cpp
	$(MPICOMPILER) $(FLAGS_VECON) -c graphsolver.cpp -o graphsolver_vecon.o $(LIBS)


isingchain_local_vecoff.o: isingchain_local.hpp qmgraph.hpp isingchain_local.cpp avx_functions.hpp
	$(MPICOMPILER) $(FLAGS_VECOFF) -c isingchain_local.cpp -o isingchain_local_vecoff.o $(LIBS)

isingchain_local_vecgcc.o: isingchain_local.hpp qmgraph.hpp isingchain_local.cpp avx_functions.hpp
	$(MPICOMPILER) $(FLAGS_VECGCC) -c isingchain_local.cpp -o isingchain_local_vecgcc.o $(LIBS)

isingchain_local_vecon.o: isingchain_local.hpp qmgraph.hpp isingchain_local.cpp avx_functions.hpp
	$(MPICOMPILER) $(FLAGS_VECON) -c isingchain_local.cpp -o isingchain_local_vecon.o $(LIBS)


qmgraph.o: qmgraph.hpp qmgraph.cpp
	$(COMPILER) $(FLAGS) -c qmgraph.cpp $(LIBS)


run: all
	export OMP_NUM_THREADS=4; \
	$(RUNCMD) ./graphsolver_vecoff.exe $(GRAPHSIZE) $(RNDSEED); \
	#$(RUNCMD) ./graphsolver_vecgcc.exe $(GRAPHSIZE) $(RNDSEED); \
	#$(RUNCMD) ./graphsolver_vecon.exe $(GRAPHSIZE) $(RNDSEED); \
	tail -n 1 diagnose*;

viz: $(GRAPHIMG_FILES)
	display *.png;

%.png: %.gv
	circo -Tpng $< -o $@

sync_euler:
	../sync_euler.sh

sync_daint:
	../sync_daint.sh

clean:
	rm -rf *.o *.exe *.gv
