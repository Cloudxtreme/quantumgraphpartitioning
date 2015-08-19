#include "qmgraph.hpp"

#include <cassert>
#include <algorithm>
#include <string>

typedef QMgraph::type_edges type_edges;
typedef QMgraph::type_count type_count;

QMgraph::QMgraph(type_count N) 
	:	N_(N)
	,	Jij_(type_edge_vec(N_*N_))
	,	partitionIndex_(0)
{
	assert(N_<=32);
}

void QMgraph::connect(type_count i, type_count j) {
	assert(i<N_ && j<N_);
	assert(i != j);			// Node cannot be connected with itself
	Jij_[i*N_+j] = 1;
	Jij_[j*N_+i] = 1;
}

type_count QMgraph::maxDegree() const {
	type_count maxDegree = 0;
	type_count degreeSum;
	for(type_count i=0; i<N_; ++i) {
		degreeSum = 0;
		for(type_count j=0; j<N_; ++j) {
			degreeSum += (Jij_[i*N_+j] > 0 ? 1 : 0);
		}
		maxDegree = std::max(maxDegree,degreeSum);
	}
	return maxDegree;
}

void QMgraph::viz(std::string graphfilename) const {
	graphfilename.append(".gv");

	FILE* outfile_ptr;
	outfile_ptr = fopen(graphfilename.c_str(),"w");
	fprintf(outfile_ptr,"# Visualization of Graph %s, size=%u, partitionIndex=%u\n\n",graphfilename.c_str(),N_,partitionIndex_);
	fprintf(outfile_ptr,"graph {\n");
	fprintf(outfile_ptr,"\tranksep=3;\n");
	fprintf(outfile_ptr,"\tratio=auto;\n");
    
    fprintf(outfile_ptr,"\n\t#TITLE\n\tlabelloc=\"t\";\n\tlabel=\"type=%s, size=%u,\\n partition index=%u\";\n",graphfilename.c_str(),N_,partitionIndex_);
	
	
	fprintf(outfile_ptr,"\n\t# NODES\n");
	for(type_count n=0; n<N_; ++n) {
		fprintf(outfile_ptr,"\tN%02u [ color=\"#000000\", fillcolor=\"%s\", shape=\"circle\", style=\"filled,solid\" ];\n",n
				,( (partitionIndex_&(type_state(1)<<n)) ? "#66ffcc" : "#ff6600"));
	}

	fprintf(outfile_ptr,"\n\t# EDGES\n");
	bool parallel;
	for(type_count j=0; j<N_; ++j) {
		for(type_count i=0; i<j; ++i) {
			// NOTE: parallel is actually antiparallel :-P
			if(Jij(i,j)) {
				parallel = ((((partitionIndex_>>i)^(partitionIndex_>>j)) & 1) == 1);
				fprintf(outfile_ptr,"\tN%02u -- N%02u [ penwidth=%u, style=\"%s\", color=\"%s\" ];\n",i,j,
						(parallel ? 2 : 3),
						(parallel ? "dashed" : "solid"),
						(parallel ? "#000000" : ( (partitionIndex_&(type_state(1)<<i)) ? "#338866" : "#884400" ))
				);
			}
		}
	}
	
	fprintf(outfile_ptr,"}");

	fclose(outfile_ptr);
	std::cout << "\n\nGraph file created as " << graphfilename << " (partition index " << partitionIndex_ << ")\n";
}

std::ostream& operator<<(std::ostream& os, const QMgraph& g) {

	// Output Connection matrix
	os << "\n\nConnection Matrix J_ij:\n\n";
	for(type_count i=0; i<g.N_; ++i) {
		os << "\t[";
		std::for_each(g.Jij_.begin()+(i*g.N_),g.Jij_.begin()+((i+1)*g.N_),[&](type_edges edge) { os << " " << edge; });
		os << " ]\n";
	}

	os << "\n\tMax Degree = " << g.maxDegree() << "\n";
	
    return os;
}
