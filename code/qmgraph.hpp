#ifndef QMGRAPH_HPP
#define QMGRAPH_HPP

#include <iostream>
#include <vector>
#include <string>

class QMgraph {

	public:

		typedef bool type_edges;
		typedef std::vector<type_edges> type_edge_vec;
		typedef unsigned short type_count;
		// IMPORTANT: MPI and SIMD implementations heavily rely on the type_state to be of size 32 !!! good up to 32 nodes
		typedef uint32_t type_state; // DON'T CHANGE

		QMgraph(type_count);
		void connect(type_count,type_count);
		type_count maxDegree() const;
		void viz(std::string) const;

		inline type_edges Jij(type_count i, type_count j) const {
			return Jij_[i*N_+j];
		}

	private:
		
		type_count N_;			// number of nodes in the graph
		type_edge_vec Jij_;		// matrix indicating connections between nodes
		type_state partitionIndex_; // Index associated with graph partitioning (bitwise)

		friend class isingChain_local;
		friend std::ostream& operator<<(std::ostream&,const QMgraph&);

};


#endif // QMGRAPH_HPP
