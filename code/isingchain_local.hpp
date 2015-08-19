#ifndef ISINGCHAIN_HPP
#define ISINGCHAIN_HPP

#include <complex>
#include <vector>
#include <cassert>
#include <chrono>
#include <string>

#include "qmgraph.hpp"

// MPI
#include <mpi.h>

// OpenMP
#include <omp.h>

// Vectorization
#include "avx_functions.hpp"	

// #define VECTORIZE 1 // Turn on and off vectorization - macro is already set by compiler

class isingChain_local {

	public: 
		
		typedef float type_scalar; // IMPORTANT: current parallelization (MPI and vectorization) assumes that type_scalar = float
		typedef QMgraph::type_state type_state;
		typedef std::complex<type_scalar> type_complex;
		typedef std::vector< type_complex, hpcse::aligned_allocator<type_complex,32> > type_coefvector; // 32 bytes = 32*8 bits = 256 for SIMD
		typedef double type_time;
		typedef QMgraph::type_count type_bitcounter;

		static const int root_id_MPI_ = 0;

		static constexpr type_scalar hbar_ = 1.;
		static constexpr type_scalar inequalityFactor_ = 1.6;

		isingChain_local(QMgraph&,unsigned,unsigned);
		~isingChain_local();
		void measure();
		void equilibrate(unsigned,type_scalar);
		void print_diagnose(std::string,std::string) const;


	private:
		
		enum Timecategory { PRE_, E0_, EP_, COMM_, NORM_, NONE_ };

		QMgraph& graph_;
		int rank_MPI_;
		int const size_MPI_;
		unsigned tag_MPI_;
		type_bitcounter N_;
		unsigned M_;
		type_scalar Tmax_;
		unsigned const dim_;
		unsigned const dim_local_MPI_;
		unsigned const offset_MPI_;
		unsigned const range_begin_MPI_;
		unsigned const range_end_MPI_;
		type_coefvector psi_;
		type_coefvector psi_tmp_;
		type_coefvector psi_other_guy_;
		type_scalar h0_;
		type_scalar A_;
		type_scalar B_;
		type_state most_probable_state_;
		type_scalar max_probability_;
		
		// Timing for benchmarking
		Timecategory timecat_;
		std::vector<type_time> timing_;
		type_time time0_, time1_;


		void normalize_();
		void communicate_flippedstates_(type_state,unsigned);

		inline type_state loc_to_glob_state_(const type_state s_local) const {
			// NOTE: removing assertions does not improve performance
			assert(s_local >= 0 && s_local < dim_local_MPI_);
			// NOTE: using offset_MPI_ rather than dim_local_MPI_ gives roughly 2% speedup of EP
			return s_local + offset_MPI_ /*dim_local_MPI_*rank_MPI_*/;
		}
		inline type_state glob_to_loc_state_(const type_state s_global) const {
			// NOTE: removing assertions does not improve performance
			assert(s_global >= range_begin_MPI_ && s_global < range_end_MPI_);
			// NOTE: using offset_MPI_ rather than dim_local_MPI_ gives roughly 2% speedup of EP
			return s_global - offset_MPI_ /*dim_local_MPI_*rank_MPI_*/;
		}

		inline void switch_to_timecat_(Timecategory new_timecat) {
			type_time time1_ = MPI_Wtime(); // stop clock
			type_time elapsed = time1_ - time0_; // calculate elapsed time
			timing_[timecat_] += elapsed; // add elapsed time to corresponding time category
			time0_ = time1_; // update time points
			timecat_ = new_timecat; // update time category
		}

};

#endif // ISINGCHAIN_HPP
