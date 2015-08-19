#include "isingchain_local.hpp"

#include <cassert>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <complex>
#include <bitset>
#include <chrono>
#include <string>

#include <mpi.h>
#include <omp.h>
#include <x86intrin.h>
#include <immintrin.h> // AVX and SSE2 (includes <emmintrin.h> automatically)


unsigned debug_MPI_rk = 1;
unsigned debug_MPI_comm = 1;

typedef isingChain_local::type_time type_time;

isingChain_local::isingChain_local(QMgraph& g, unsigned rank, unsigned nprocs) 
	:	graph_(g)
	,	rank_MPI_(rank)
	,	size_MPI_(nprocs)
	,	tag_MPI_(-1)
	,	N_(graph_.N_)
	,	M_(0)
	,	Tmax_(0)
	,	dim_(1<<N_)
	,	dim_local_MPI_(dim_/size_MPI_)
	,	offset_MPI_(rank_MPI_*dim_local_MPI_)
	,	range_begin_MPI_(rank_MPI_*dim_local_MPI_)
	,	range_end_MPI_(range_begin_MPI_+dim_local_MPI_)
	,	psi_(type_coefvector(dim_local_MPI_))
	,	psi_tmp_(type_coefvector(dim_local_MPI_))
	,	psi_other_guy_(type_coefvector(dim_local_MPI_))
	,	h0_(1.)
	,	A_(h0_)
	,	B_(h0_)
	,	most_probable_state_(dim_)
	,	max_probability_(0.)
	,	timecat_(NONE_)
	,	timing_(std::vector<type_time>(NONE_+1,0.))
	,	time0_(MPI_Wtime())
	,	time1_(time0_)
{
	
	switch_to_timecat_(NONE_);

	// assert( nprocs == 2^k ) - Number of processors must be power of 2
	assert(__builtin_popcount(nprocs) == 1);

	// Initialization of coefficient vector psi
	// with ground state of H0
	for(type_state s_global=range_begin_MPI_; s_global<range_end_MPI_; ++s_global) {
		bool even_downs = (N_ - __builtin_popcount(s_global))%2; // DOWN = N - UP
		psi_[glob_to_loc_state_(s_global)] = ( even_downs ? +1 : -1 );
	}
	normalize_();

	psi_other_guy_ = psi_tmp_ = psi_; // set all coefficient vectors equal

	MPI_Barrier(MPI_COMM_WORLD);

}

isingChain_local::~isingChain_local() {}


void isingChain_local::measure() {
		
	switch_to_timecat_(NONE_);
	
	MPI_Barrier(MPI_COMM_WORLD);

	type_scalar const pAvg = 1./sqrt(dim_);
	
	std::cout.precision(15);
	if(rank_MPI_ == root_id_MPI_) {
		std::cout << "\n\nMost probable Basis Elements:";
		std::cout << "\n(average probability = " << pAvg << ")\n";
	}

	// Enforce serial output: this is just for nicer logging
	for(int p_id=0; p_id<size_MPI_; ++p_id) {
		if(rank_MPI_ == p_id) {
		
			// Output probabilities of nodes:
			for(type_state s_local=0; s_local<dim_local_MPI_; ++s_local) {
			
				type_scalar p = std::norm(psi_[s_local]); // probability of basis state

				if(p > max_probability_) {
					max_probability_ = p;
					most_probable_state_ = loc_to_glob_state_(s_local);
				}
			
				// std::cout << "\np=" << p;
				if((p>2.*pAvg) || ((debug_MPI_rk>=1) && (dim_local_MPI_<=16))) {
					std::cout << "\n\t[";
					type_state s_global = loc_to_glob_state_(s_local); // this is the global index which encodes state s
					for(unsigned r=0; r<N_; ++r) {
						// check if r-th bit spin is up
						std::cout << " " << ( (type_state(s_global&(1<<r)) == type_state(1<<r)) ? 1 : 0 );
					}
					std::cout << " ]\t\tp = " << std::fixed << p << std::flush;
				}
			
			}
		
		}

		MPI_Barrier(MPI_COMM_WORLD); // Enforce serial output:
	}	


	// Write most probable state to root process and update partitionIndex_ in graph_
	std::vector<type_scalar> max_probability_vec(size_MPI_);
	std::vector<type_state> most_probable_state_vec(size_MPI_);
	
	MPI_Gather(&max_probability_, 1, MPI_FLOAT, &max_probability_vec[0], 1, MPI_FLOAT, isingChain_local::root_id_MPI_, MPI_COMM_WORLD);
	MPI_Gather(&most_probable_state_, 1, MPI_UNSIGNED, &most_probable_state_vec[0], 1, MPI_UNSIGNED, isingChain_local::root_id_MPI_, MPI_COMM_WORLD);
		
	// Create and write output file
	if(rank_MPI_ == isingChain_local::root_id_MPI_) {
		// get the global maximum probability and the corresponding state
		// write to root process
		for(int loc=0; loc<size_MPI_; ++loc) {
			if(max_probability_vec[loc] > max_probability_) {
				most_probable_state_ = most_probable_state_vec[loc];
				max_probability_ = max_probability_vec[loc];
			}
		}
	}
	graph_.partitionIndex_ = most_probable_state_;

	switch_to_timecat_(NONE_);

}

void isingChain_local::equilibrate(unsigned M, type_scalar T) {

	switch_to_timecat_(PRE_);
	
	M_ = M;
	Tmax_ = T;

	B_ = h0_;
	A_ = inequalityFactor_ * B_/8. * std::min(type_bitcounter(2*graph_.maxDegree()),N_); // 2*short is still short

	type_scalar const tau = T/M;

	type_scalar alpha_k, beta_k_ij_connected, beta_k_ij_disconnected, gamma_k;
	type_complex cos_gamma, isin_gamma, expalpha, expbeta, expbeta_connected, expbeta_disconnected;
	type_state s_flip, start_flip;

	for(unsigned k=1; k<=M; ++k) {

		switch_to_timecat_(PRE_);
	
		/////////////////////////////////////////////////////
		// Apply hamiltonian H(t_k)
		/////////////////////////////////////////////////////

		// Precomputing constants, exponential & trigonometric functions
		alpha_k = -1./hbar_ * tau * type_scalar(k)/M * A_;
		beta_k_ij_connected		= -1./hbar_ * tau * type_scalar(k)/M * (B_-A_);
		beta_k_ij_disconnected	= -1./hbar_ * tau * type_scalar(k)/M * (  -A_);
		gamma_k = -1./hbar_ * tau * (type_scalar(k)/M-1) * h0_;

		expalpha = std::exp(type_complex(0,alpha_k));
		expbeta_connected		= std::exp(type_complex(0,beta_k_ij_connected));
		expbeta_disconnected	= std::exp(type_complex(0,beta_k_ij_disconnected));
		cos_gamma	= type_complex(std::cos(gamma_k),0);
		isin_gamma	= type_complex(0,std::sin(gamma_k));

#if VECTORIZE > 0
			type_scalar cos_gamma_realpart = cos_gamma.real();
			type_scalar isin_gamma_imagpart = isin_gamma.imag();
#endif // VECTORIZE

		/////////////////////////////////////////////////////
		// E_0	
		switch_to_timecat_(E0_);
		for(type_bitcounter ri=0; ri<N_; ++ri) {

				// NOTE: psi_ = psi_MPI_out_
				// NOTE: psi_other_guy_ = psi_MPI_in_

				// diagonal
				// LOCAL OPERATIONS
				#pragma omp parallel for shared(cos_gamma)
				for(type_state s_local_vecindex=0; s_local_vecindex<dim_local_MPI_; s_local_vecindex+=4) {
					// IMPORTANT: current implementation of vectorization assumes that type_scalar = float
#if VECTORIZE > 0
						__m256 cg = _mm256_set1_ps(cos_gamma_realpart);
						// +=4 for complex float = 8 single float, reinterpret complex<float> as float[2]
						__m256 psi = _mm256_load_ps(reinterpret_cast<type_scalar*>(&psi_[s_local_vecindex])); // Loading from memory
						__m256 psitmp = hpcseavx::avx_multiply_float_real_(cg,psi);
						_mm256_stream_ps(reinterpret_cast<type_scalar*>(&psi_tmp_[s_local_vecindex]),psitmp); // Writing to memory, bypassing cache
#else 
						// <==> psi_tmp_[s_local] = cos_gamma * psi_[s_local];
						psi_tmp_[s_local_vecindex] = cos_gamma * psi_[s_local_vecindex];
						psi_tmp_[s_local_vecindex+1] = cos_gamma * psi_[s_local_vecindex+1];
						psi_tmp_[s_local_vecindex+2] = cos_gamma * psi_[s_local_vecindex+2];
						psi_tmp_[s_local_vecindex+3] = cos_gamma * psi_[s_local_vecindex+3];
#endif // VECTORIZE
				}
			
				// off-diagonal
				// check if the flip is local to MPI-unit
				start_flip = range_begin_MPI_^(1<<ri); // flipping spin of start index of current MPI block
				bool localflip = (start_flip >= range_begin_MPI_) && (start_flip < range_end_MPI_);
				if(localflip) {
					// no transfer of data required
					#pragma omp parallel for private(s_flip) shared(isin_gamma)
					for(type_state s_local=0; s_local<dim_local_MPI_; ++s_local) {
						s_flip = s_local^(1<<ri);
						// NOTE: cannot be conveniently vectorized, since s_flip != s_local
						psi_tmp_[s_flip] += isin_gamma * psi_[s_local]; // use local data in psi_
					}
				} else {
					// write state coefficients from psi_ to psi_other_guy_
					switch_to_timecat_(COMM_);
					communicate_flippedstates_(ri,(k*N_)+ri); // unique communication tag k*N + ri
					switch_to_timecat_(E0_);
				
					#pragma omp parallel for shared(isin_gamma)
					for(type_state s_local_vecindex=0; s_local_vecindex<dim_local_MPI_; s_local_vecindex+=4) {
						// use transferred data in psi_other_guy_
						// communication implicitly takes care of s_flip, therefore no flipping required
#if VECTORIZE > 0
							__m256 sg = _mm256_set1_ps(isin_gamma_imagpart);
							// +=4 for complex float = 8 single float, reinterpret complex<float> as float[2] 
							__m256 psi = _mm256_load_ps(reinterpret_cast<type_scalar*>(&psi_other_guy_[s_local_vecindex])); // Loading from memory
							__m256 psitmp = _mm256_load_ps(reinterpret_cast<type_scalar*>(&psi_tmp_[s_local_vecindex])); // Loading from memory
							psi = hpcseavx::avx_multiply_float_imag_(sg,psi);
							psitmp = _mm256_add_ps(psitmp,psi);
							_mm256_stream_ps(reinterpret_cast<type_scalar*>(&psi_tmp_[s_local_vecindex]),psitmp); // Writing to memory, bypassing cache
#else
							// <==> psi_tmp_[s_local] += isin_gamma * psi_other_guy_[s_local];
							psi_tmp_[s_local_vecindex] += isin_gamma * psi_other_guy_[s_local_vecindex];
							psi_tmp_[s_local_vecindex+1] += isin_gamma * psi_other_guy_[s_local_vecindex+1];
							psi_tmp_[s_local_vecindex+2] += isin_gamma * psi_other_guy_[s_local_vecindex+2];
							psi_tmp_[s_local_vecindex+3] += isin_gamma * psi_other_guy_[s_local_vecindex+3];
#endif // VECTORIZE

					}
				}

			std::swap(psi_,psi_tmp_);
		}



		/////////////////////////////////////////////////////
		// E_P
		switch_to_timecat_(EP_);
		#pragma omp parallel firstprivate(expbeta)
		{
			// Initialized after spawning threads - makes vectors thread-private
			// automatically private since declared within parallel region

			std::vector<type_state> s_global(4); // private

#if VECTORIZE > 0
			type_coefvector exp_factor(4); // private
			type_coefvector expalpha_vec(4); // private
			std::fill(expalpha_vec.begin(), expalpha_vec.end(), expalpha);
			type_coefvector expbeta_vec(4); // private
#else
			type_complex exp_factor;
			bool parallel;
#endif // VECTORIZE

			#pragma omp for
			for(type_state s_local_vecindex=0; s_local_vecindex<dim_local_MPI_; s_local_vecindex+=4) {
			
				s_global[0] = loc_to_glob_state_(s_local_vecindex);
				s_global[1] = loc_to_glob_state_(s_local_vecindex+1);
				s_global[2] = loc_to_glob_state_(s_local_vecindex+2);
				s_global[3] = loc_to_glob_state_(s_local_vecindex+3);

#if VECTORIZE > 0
				__m256 psi_avx = _mm256_load_ps(reinterpret_cast<type_scalar*>(&psi_[s_local_vecindex]));
				__m256 expalpha_avx, expbeta_avx, expfac_avx;
#endif // VECTORIZE
				
				for(type_bitcounter ri=0; ri<N_; ++ri) {
					for(type_bitcounter rj=0; rj<N_; ++rj) {

						// LOCAL OPERATIONS
						expbeta = ( graph_.Jij(ri,rj) == 1 ? expbeta_connected : expbeta_disconnected );
#if VECTORIZE > 0
					
						// Cannot use set1 since complex numbers
						std::fill(expbeta_vec.begin(), expbeta_vec.end(), expbeta);

						// SSE2
						//	cast unsigned int to signed int and load to sse2 register
						__m128i s_sse = _mm_load_si128( (__m128i*)&s_global[0] );

						//	bitshift with ri, rj:
						//	shifting r-th bit to sign bit:
						//	- shift right by r to get to least significant bit
						//	- shift left by 31 to move least significant bit to sign bit
						//	<==> shift left by 31-r = 0x1F-r
						__m128i shift_ri_sse = _mm_set1_epi32(0x1F-ri); // 0x1F = 31: shift relevant bit to sign bit
						__m128i shift_rj_sse = _mm_set1_epi32(0x1F-rj); // 0x1F = 31: shift relevant bit to sign bit
						shift_ri_sse = _mm_sll_epi32(s_sse,shift_ri_sse);
						shift_rj_sse = _mm_sll_epi32(s_sse,shift_rj_sse);

						// Check if bits (i.e. spins) are parallel
						s_sse = _mm_xor_si128(shift_ri_sse,shift_rj_sse);
						__m128 mask_float_sse = _mm_castsi128_ps(s_sse); // cast mask to float

						// AVX
						// cast mask to avx and duplicate entries for real and complex floats
						__m256 mask_avx = _mm256_castps128_ps256(mask_float_sse);
						mask_avx = _mm256_insertf128_ps(mask_avx,mask_float_sse,0x01);
						__m256 mask_lo_avx = _mm256_unpacklo_ps(mask_avx,mask_avx);
						__m256 mask_hi_avx = _mm256_unpackhi_ps(mask_avx,mask_avx);
						mask_avx = _mm256_blend_ps(mask_lo_avx,mask_hi_avx,0xF0);

						expalpha_avx = _mm256_load_ps(reinterpret_cast<type_scalar*>(&expalpha_vec[0]));
						expbeta_avx = _mm256_load_ps(reinterpret_cast<type_scalar*>(&expbeta_vec[0]));

						expfac_avx = _mm256_blendv_ps(expbeta_avx,expalpha_avx,mask_avx);
						psi_avx = hpcseavx::avx_multiply_float_complex_(expfac_avx,psi_avx);
#else
						for(unsigned short k=0; k<4; ++k) {
							parallel = (((s_global[k]>>ri)^(s_global[k]>>rj)) & 1) == 1;
							exp_factor = ( parallel ? expalpha : expbeta );
							psi_[s_local_vecindex+k] *= exp_factor;
						}
#endif // VECTORIZE
					}
				}
#if VECTORIZE > 0
				_mm256_stream_ps(reinterpret_cast<type_scalar*>(&psi_[s_local_vecindex]),psi_avx);
				// NB: store is faster for small graphs, stream is faster for large graps
#endif // VECTORIZE
			} // end of parallel for
		} // end of parallel region
	}
		
	switch_to_timecat_(NORM_);
	normalize_();
	switch_to_timecat_(COMM_);
	MPI_Barrier(MPI_COMM_WORLD);
	switch_to_timecat_(NONE_);		
}


void isingChain_local::normalize_() {

	// MPI_Barrier(MPI_COMM_WORLD);

	type_scalar norm2 = 0;
	
	// No parallelization needed, only < .1% of time used for normalization
	for(auto p : psi_) norm2 += std::norm(p);

	switch_to_timecat_(COMM_);
	MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	switch_to_timecat_(NORM_);

	assert(norm2 > 0.); // avoid division by zero
	type_scalar c = 1./sqrt(norm2); // normalization factor

	std::for_each(psi_.begin(),psi_.end(),[&](type_complex& x) { x *= c; }); // normalize
	
	//MPI_Barrier(MPI_COMM_WORLD);

}


void isingChain_local::communicate_flippedstates_(type_state const r, unsigned const comm_tag) {
	
	//MPI_Barrier(MPI_COMM_WORLD);
	
	// calculate other guy
	type_state s_start_flipped = (range_begin_MPI_^(1<<r));
	assert(s_start_flipped%dim_local_MPI_ == 0);
	int other_guy = s_start_flipped/dim_local_MPI_;

	MPI_Status status;

	// Sendrecv uses a blocking send / receive	
	int msg = MPI_Sendrecv(&psi_.front(), 2*dim_local_MPI_, MPI_FLOAT,
			other_guy, comm_tag,
			&psi_other_guy_.front(), 2*dim_local_MPI_, MPI_FLOAT,
			other_guy, comm_tag,
			MPI_COMM_WORLD, &status); // MPI_STATUS_IGNORE);

	assert(msg == 0); // check if there was a major communication problem

	//MPI_Barrier(MPI_COMM_WORLD);
}

void isingChain_local::print_diagnose(std::string prefix, std::string suffix) const {

	type_time time_PRE = timing_[PRE_];
	type_time time_E0 = timing_[E0_];
	type_time time_EP = timing_[EP_];
	type_time time_norm = timing_[NORM_];
	type_time time_comm = timing_[COMM_];

	type_time time_PRE_tot, time_E0_tot, time_EP_tot, time_norm_tot, time_comm_tot;

	MPI_Reduce(&time_PRE, &time_PRE_tot, 1, MPI_DOUBLE, MPI_SUM, isingChain_local::root_id_MPI_, MPI_COMM_WORLD);
	MPI_Reduce(&time_E0, &time_E0_tot, 1, MPI_DOUBLE, MPI_SUM, isingChain_local::root_id_MPI_, MPI_COMM_WORLD);
	MPI_Reduce(&time_EP, &time_EP_tot, 1, MPI_DOUBLE, MPI_SUM, isingChain_local::root_id_MPI_, MPI_COMM_WORLD);
	MPI_Reduce(&time_norm, &time_norm_tot, 1, MPI_DOUBLE, MPI_SUM, isingChain_local::root_id_MPI_, MPI_COMM_WORLD);
	MPI_Reduce(&time_comm, &time_comm_tot, 1, MPI_DOUBLE, MPI_SUM, isingChain_local::root_id_MPI_, MPI_COMM_WORLD);

	// Create and write output file
	if(rank_MPI_ == isingChain_local::root_id_MPI_) {

		// Get current timestamp
		auto now = std::chrono::system_clock::now();
		std::time_t now_c = std::chrono::system_clock::to_time_t(now);
		struct tm* loctime = std::localtime(&now_c);

		// Constructing filename
		std::string filename("diagnose_");
		filename.append(prefix); filename.append("_");
		filename.append("r"); filename.append(std::to_string(size_MPI_));
		filename.append("t"); filename.append(std::to_string(omp_get_max_threads()));
		filename.append("v"); filename.append(std::to_string(VECTORIZE));
		filename.append("s"); filename.append(std::to_string(N_));
		filename.append("_");
		std::stringstream sstr_year, sstr_mon, sstr_mday, sstr_hour, sstr_min, sstr_sec;
		sstr_year << std::setw(4) << std::setfill('0') << 1900+loctime->tm_year; filename.append(sstr_year.str());
		sstr_mon << std::setw(2) << std::setfill('0') << 1+loctime->tm_mon; filename.append(sstr_mon.str());
		sstr_mday << std::setw(2) << std::setfill('0') << loctime->tm_mday << "_"; filename.append(sstr_mday.str());
		sstr_hour << std::setw(2) << std::setfill('0') << loctime->tm_hour << "-"; filename.append(sstr_hour.str());
		sstr_min << std::setw(2) << std::setfill('0') << loctime->tm_min << "-"; filename.append(sstr_min.str());
		sstr_sec << std::setw(2) << std::setfill('0') << loctime->tm_sec; filename.append(sstr_sec.str());
		filename.append("_"); filename.append(suffix);
		filename.append(".csv");

		// NB: no convenient MPI-function for getting MPI ranks/processes per node
		// we introduced our own environment variable to be consistent 
		// between program invocation and output file
		// run mpirun with -npernode=$DUDE_MPI_RANKS_PER_NODE
		// (default = 1)
		std::string mpi_nnodes_str;
		if(const char* env_p = std::getenv("DUDE_MPI_RANKS_PER_NODE")) {
			mpi_nnodes_str = std::string(env_p);
		} else {
			mpi_nnodes_str = "1";
		}
	
		FILE* outfile_ptr;
		outfile_ptr = fopen(filename.c_str(),"w");
		unsigned timestamp = std::chrono::system_clock::now().time_since_epoch() / std::chrono::seconds(1);
		fprintf(outfile_ptr,"# Timestamp & Code Version:\n# Timestamp, MPI_FLAG, OMP_FLAG, VEC_FLAG\n%u, %u, %u, %i\n#\n",timestamp,(size_MPI_>1 ? 1 : 0),(omp_get_max_threads()>1 ? 1 : 0),int(VECTORIZE));
		fprintf(outfile_ptr,"# Graph Parameters:\n# Graph Size, Tmax, M\n%u, %f, %u\n#\n",graph_.N_,Tmax_,M_);
		fprintf(outfile_ptr,"# Measurement (most likely state):\n# StateIndex, Probability\n%u, %.10f\n#\n",most_probable_state_,max_probability_);
		fprintf(outfile_ptr,"# Parallellization Parameters:\n# MPI_ranks, MPI_nnodes, OMP_threads\n%u, %s, %i\n#\n",size_MPI_,mpi_nnodes_str.c_str(),omp_get_max_threads());
		fprintf(outfile_ptr,"# Timing in seconds\n# PRE, E0, EP, NORM, COMM\n%.20f, %.20f, %.20f, %.20f, %.20f",
				time_PRE_tot, time_E0_tot, time_EP_tot, time_norm_tot, time_comm_tot);
		fclose (outfile_ptr);

	}

}
