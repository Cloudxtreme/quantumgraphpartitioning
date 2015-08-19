#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <complex>
#include <algorithm>

#include <mpi.h>
#include <omp.h>

#include "isingchain_local.hpp"
#include "qmgraph.hpp"


#define N_ARGS 2

int main(int argc, char* argv[]) {

	assert(argc >= N_ARGS);
	unsigned nNodes = atoi(argv[1]);
	
	int rndseed;
	
	if(argc >= 3) {
		rndseed = atoi(argv[2]);
	} else {
		rndseed = 42;
	}
	
	typedef isingChain_local::type_scalar type_scalar;

	int size_MPI, rank_MPI;
	int provided_MPI_OMP_hybrid;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided_MPI_OMP_hybrid);
	assert(provided_MPI_OMP_hybrid >= MPI_THREAD_FUNNELED);
	MPI_Comm_size(MPI_COMM_WORLD,&size_MPI); // total number of MPI-processes
	MPI_Comm_rank(MPI_COMM_WORLD,&rank_MPI); // rank of current process
	
	// /////////////////////////////////////////////////////////////
	// // SIMULATION PARAMETERS PHYSICAL
	// /////////////////////////////////////////////////////////////
	type_scalar Tmax = 10.;
	type_scalar dt = 1e-2;
	unsigned nsteps = std::round(Tmax/dt);
	// /////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////
	// SIMULATION PARAMETERS UNPHYSICAL -> 2 timesteps
	/////////////////////////////////////////////////////////////
	// type_scalar Tmax = 2.;
	// type_scalar dt = 1;
	// unsigned nsteps = std::round(Tmax/dt);
	/////////////////////////////////////////////////////////////


	//unsigned nNodes = 10;
	// NOTE: nEdges only affects random graph
	// Choosing a sensible number of edges for graph
	unsigned nEdges = std::max(1u,nNodes*(nNodes-1)/4); 

	// graph must have at least one edge, and less than the maximum possible number of edges
	assert(nEdges > 0 && nEdges <= nNodes*(nNodes-1)/2);
	srand(rndseed);

	std::string vectorize_str;
	switch(int(VECTORIZE)) {
		case 0:
			vectorize_str = "OFF";
			break;
		case 1:
			vectorize_str = "ON";
			break;
		case -1:
			vectorize_str = "GCC";
			break;
	}

	if(rank_MPI==isingChain_local::root_id_MPI_) {
		std::cout << "\n\t============================================"
					<< "\n\t========== Simulation Parameters: =========="
					<< "\n\t============================================"
					<< "\n\t\tGraph Nodes:      " << nNodes 
					<< "\n\t\tGraph Edges:      " << nEdges
					<< "\n\t\tTmax:             " << Tmax 
					<< "\n\t\ttimesteps:        " << nsteps
					<< "\n\t\tdt:               " << dt
					<< "\n\t\trandom seed:      " << rndseed
					<< "\n\t============================================"
					<< "\n\t\tMPI processes:    " << size_MPI
					<< "\n\t\tOMP max_threads:  " << omp_get_max_threads()
					<< "\n\t\tSIMD:             " << vectorize_str
					<< "\n\t============================================\n";
	}


	/////////////////////////////////////////////////////////////
	// TESTGRAPH SUBCLUSTERS: 2 subclusters of size nNodes/2
	/////////////////////////////////////////////////////////////

	if(rndseed == -1) {
		
		if(rank_MPI==isingChain_local::root_id_MPI_) std::cout << "\n\n\n\n========== FINDING PARTITION OF TEST GRAPH WITH 2 SUBCLUSTERS: ==========" << std::flush;

		QMgraph testgraph_sub2(nNodes);

		// First Subcluster
		for(unsigned i=0; i<nNodes/2; ++i) {
			for(unsigned j=(i+1); j<nNodes/2; ++j) {
				testgraph_sub2.connect(i,j);
			}
		}

		// Second Subcluster
		for(unsigned i=nNodes/2; i<nNodes; ++i) {
			for(unsigned j=(i+1); j<nNodes; ++j) {
				testgraph_sub2.connect(i,j);
			}
		}

		// Connect Subclusters by single edge
		testgraph_sub2.connect(0,nNodes-1);
		
		if(rank_MPI==isingChain_local::root_id_MPI_) std::cout << testgraph_sub2;

		// Find Partitioning
		isingChain_local ic_test_sub2(testgraph_sub2,rank_MPI,size_MPI);

		ic_test_sub2.equilibrate(nsteps,Tmax);
		ic_test_sub2.measure();
		ic_test_sub2.print_diagnose("subtwo","");
		if(rank_MPI==isingChain_local::root_id_MPI_) testgraph_sub2.viz("subtwo");

		MPI_Barrier(MPI_COMM_WORLD);
	}
	//////////////////////////////////////////////////////////////////////////////////

	
	/////////////////////////////////////////////////////////////
	// TESTGRAPH CIRCLE
	/////////////////////////////////////////////////////////////

	if(rndseed == -2) {
		if(rank_MPI==isingChain_local::root_id_MPI_)	std::cout << "\n\n\n\n========== FINDING PARTITION OF TEST GRAPH CIRCLE: ==========" << std::flush;

		QMgraph testgraph_circle(nNodes);

		// Circular Connections
		for(unsigned i=0; i<nNodes; ++i) {
			testgraph_circle.connect(i,(i+1)%nNodes);
		}

		if(rank_MPI==isingChain_local::root_id_MPI_)	std::cout << testgraph_circle;

		// Find Partitioning
		isingChain_local ic_test_circle(testgraph_circle,rank_MPI,size_MPI);
		ic_test_circle.equilibrate(nsteps,Tmax);
		ic_test_circle.measure();
		ic_test_circle.print_diagnose("circ","");
		if(rank_MPI==isingChain_local::root_id_MPI_) testgraph_circle.viz("circ");
		
		MPI_Barrier(MPI_COMM_WORLD);
	}
	//////////////////////////////////////////////////////////////////////////////////


	/////////////////////////////////////////////////////////////
	// RANDOM GRAPH with nEdges edges
	/////////////////////////////////////////////////////////////

	if((rndseed != -1) && (rndseed != -2)) {
		if(rank_MPI==isingChain_local::root_id_MPI_)	std::cout << "\n\n\n\n========== FINDING PARTITION OF RANDOM GRAPH: ==========" << std::flush;
		
		QMgraph randomgraph(nNodes);

		// First Subcluster
		unsigned i, j;
		for(unsigned c=0; c<nEdges; ++c) {
			i = rand()%nNodes;
			do {
				j = rand()%nNodes;
			} while(j == i); // don't connect nodes with themselves
			randomgraph.connect(i,j);
		}
		if(rank_MPI==isingChain_local::root_id_MPI_)	std::cout << randomgraph;

		// Find Partitioning
		isingChain_local ic_random(randomgraph,rank_MPI,size_MPI);
		ic_random.equilibrate(nsteps,Tmax);
		ic_random.measure();
		ic_random.print_diagnose("rand"+std::to_string(rndseed),"");
		if(rank_MPI==isingChain_local::root_id_MPI_) randomgraph.viz("rand"+std::to_string(rndseed));

		MPI_Barrier(MPI_COMM_WORLD);
	}
	//////////////////////////////////////////////////////////////////////////////////
	
	if(rank_MPI==isingChain_local::root_id_MPI_)	std::cout << "\n\n\n";

	MPI_Finalize();

	return 0;

}
