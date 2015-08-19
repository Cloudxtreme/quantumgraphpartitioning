// Vectorization
#include "aligned_allocator.hpp"
#include <x86intrin.h>
#include <immintrin.h>

namespace hpcseavx {
		
	// PRE: all vectors aligned, 
	//		real_c = [r1,r1,...,r4,r4]
	//		vec = [v1r,v1i,...,v4r,v4i]
	//		component-wise multiplication
	// POST: returns [r1*v1r,r1*v1i,...,r4*v4r,r4*v4i]
	inline __m256 avx_multiply_float_real_(const __m256& real_c, const __m256& vec) {
		return _mm256_mul_ps(real_c,vec);
	}

	// PRE: all vectors aligned, 
	//		imag_c = [i1,i1,...,i4,i4]
	//		vec = [v1r,v1i,...,v4r,v4i]
	//		component-wise multiplication
	// POST: returns [-i1*v1i,i1*v1r,...,-i4*v4i,i4*v4r]
	inline __m256 avx_multiply_float_imag_(const __m256& imag_c, const __m256& vec) {
		static const __m256 zero = _mm256_setzero_ps();
		__m256 vec1 = _mm256_mul_ps(imag_c,vec);
		vec1 = _mm256_permute_ps(vec1,0xB1);
		return _mm256_addsub_ps(zero,vec1);
	}

	// PRE: all vectors aligned, 
	//		vecA = [A1r,A1i,...,A4r,A4i]
	//		vecB = [B1r,B1i,...,B4r,B4i]
	//		full complex multiplication
	// POST: returns [A1r*B1r-A1i*B1i,A1r*B1i+A1i*B1r,...,A4r*B4r-A4i*B4i,A4r*B4i+A4i*B4r]
	// NOT NEEDED
	/****************************************************************
	 * This technique for efficient SIMD complex-complex multiplication was found at
	 *			https://software.intel.com/file/1000
	*****************************************************************/
	inline __m256 avx_multiply_float_complex_(const __m256& vecA, const __m256& vecB) {
		__m256 vec1 = _mm256_moveldup_ps(vecB);
		__m256 vec2 = _mm256_movehdup_ps(vecB);
		vec1 = _mm256_mul_ps(vecA,vec1);
		vec2 = _mm256_mul_ps(vecA,vec2); 
		vec2 = _mm256_permute_ps(vec2,0xB1); 
		return _mm256_addsub_ps(vec1,vec2);
	}

}
