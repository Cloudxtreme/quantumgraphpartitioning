HPCSE II final project
======================

Graph Partitioning using Adiabatic Optimization on an Ising Chain
-----------------------------------------------------------------

Authors: Pascal Iselin & Matthias Untergassmair 
Lecture: ETH Zürich HighPerformanceComputing II Project, spring semester 2015

Note: this public repository only holds a reduced version of the original project

* `make run` in the `code` directory for compiling and running the program with default parameters
* System requirements: GCC compiler, support for AVX, OpenMP, MPI
* Setting parallelization parameters
	- MPI: run the program prefixed by `mpirun -np P`, where `P` is the number of MPI processes to run
	- OpenMP: set the corresponding environment variable via export `OMP_NUM_THREADS=N` where `N` is the number of OpenMP threads
	- Vectorization: the macro `VECTORIZE` must be set at compile time via `-D VECTORIZE=0` (SIMD off) or `-D VECTORIZE=1` (SIMD on).
* visualize graph and solution: after having run a simulation, `make viz` draws the partitioned graph using graphviz ('www.graphviz.org'). Make sure it is installed on the system.
* Input Arguments:
	- First input argument `N`: must be an integer between 2 and 32 which determines the number of nodes in the graph
		NOTE: runtime and memory scale exponentially with the number of nodes!
	- Second input argument `R` (optional): is the random seed for the graph. In the special case of `R` equal `-1` the graph consists of two maximally connected subgraphs with only one connection. In the special case of `R` equal `-2` the graph is a circular graph. Default random seed is `42`.
* benchmark.db is a sqlite3 database that summarizes the benchmark results obtained from timing our runs on the Piz Daint Supercomputer at the CSCS Lugano, the `plots` directory holds the associated graphs
