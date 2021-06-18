/*
 * Copyright (c) 2017-2021 Genome Research Ltd.
 * Author(s): James Bonfield
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    1. Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the names Genome Research Ltd and Wellcome Trust Sanger
 *       Institute nor the names of its contributors may be used to endorse
 *       or promote products derived from this software without specific
 *       prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY GENOME RESEARCH LTD AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL GENOME RESEARCH
 * LTD OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#if defined(HAVE_SSE4_1) && defined(HAVE_SSSE3) && defined(HAVE_POPCNT)

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <x86intrin.h>

#ifndef NO_THREADS
#include <pthread.h>
#endif

#include "rANS_word.h"
#include "rANS_static4x16.h"
#include "rANS_static16_int.h"
#include "varint.h"
#include "pack.h"
#include "rle.h"
#include "utils.h"

/* Uses: SSE, SSE2, SSSE3, SSE4.1 and POPCNT
SSE:
_mm_movemask_ps

SSE2:
    _mm_load_si128 _mm_store_si128
    _mm_set_epi32  _mm_set1_epi32
    _mm_and_si128  _mm_or_si128
    _mm_srli_epi32 _mm_slli_epi32
    _mm_add_epi32
    _mm_packus_epi32
    _mm_andnot_si128
    _mm_cmpeq_epi32

SSSE3:
    _mm_shuffle_epi8

SSE4.1:
    _mm_mullo_epi32
    _mm_packus_epi32
    _mm_max_epu32
    _mm_cvtepu16_epi32
    _mm_blendv_epi8

POPCNT:
    _mm_popcnt_u32
 */


#define NX 32

#define LOAD128(a,b)					\
    __m128i a##1 = _mm_load_si128((__m128i *)&b[0]);	\
    __m128i a##2 = _mm_load_si128((__m128i *)&b[4]);	\
    __m128i a##3 = _mm_load_si128((__m128i *)&b[8]);	\
    __m128i a##4 = _mm_load_si128((__m128i *)&b[12]);	\
    __m128i a##5 = _mm_load_si128((__m128i *)&b[16]);	\
    __m128i a##6 = _mm_load_si128((__m128i *)&b[20]);	\
    __m128i a##7 = _mm_load_si128((__m128i *)&b[24]);	\
    __m128i a##8 = _mm_load_si128((__m128i *)&b[28]);

#define STORE128(a,b)					\
    _mm_store_si128((__m128i *)&b[ 0], a##1);		\
    _mm_store_si128((__m128i *)&b[ 4], a##2);		\
    _mm_store_si128((__m128i *)&b[ 8], a##3);		\
    _mm_store_si128((__m128i *)&b[12], a##4);		\
    _mm_store_si128((__m128i *)&b[16], a##5);		\
    _mm_store_si128((__m128i *)&b[20], a##6);		\
    _mm_store_si128((__m128i *)&b[24], a##7);		\
    _mm_store_si128((__m128i *)&b[28], a##8);

static inline __m128i _mm_i32gather_epi32x(int *b, __m128i idx, int size) {
    int c[4] __attribute__((aligned(32)));
    _mm_store_si128((__m128i *)c, idx);
    return _mm_set_epi32(b[c[3]], b[c[2]], b[c[1]], b[c[0]]);
}

unsigned char *rans_uncompress_O0_32x16_sse4(unsigned char *in,
					     unsigned int in_size,
					     unsigned char *out,
					     unsigned int out_sz) {
    if (in_size < 16) // 4-states at least
	return NULL;

    if (out_sz >= INT_MAX)
	return NULL; // protect against some overflow cases

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    if (out_sz > 100000)
	return NULL;
#endif

    /* Load in the static tables */
    unsigned char *cp = in, *out_free = NULL;
    unsigned char *cp_end = in + in_size - 8; // within 8 => be extra safe
    int i, j;
    unsigned int x, y;
    uint8_t  ssym [TOTFREQ+64]; // faster to use 16-bit on clang
    uint32_t s3[TOTFREQ] __attribute__((aligned(32))); // For TF_SHIFT <= 12

    if (!out)
	out_free = out = malloc(out_sz);
    if (!out)
	return NULL;

    // Precompute reverse lookup of frequency.
    uint32_t F[256] = {0}, fsum;
    int fsz = decode_freq(cp, cp_end, F, &fsum);
    if (!fsz)
	goto err;
    cp += fsz;

    normalise_freq_shift(F, fsum, TOTFREQ);

    // Build symbols; fixme, do as part of decode, see the _d variant
    for (j = x = 0; j < 256; j++) {
	if (F[j]) {
	    if (F[j] > TOTFREQ - x)
		goto err;
	    for (y = 0; y < F[j]; y++) {
		ssym [y + x] = j;
		s3[y+x] = (((uint32_t)F[j])<<(TF_SHIFT+8))|(y<<8)|j;
	    }
	    x += F[j];
	}
    }

    if (x != TOTFREQ)
	goto err;

    if (cp+16 > cp_end+8)
	goto err;

    int z;
    RansState R[NX] __attribute__((aligned(32)));
    for (z = 0; z < NX; z++) {
	RansDecInit(&R[z], &cp);
	if (R[z] < RANS_BYTE_L)
	    goto err;
    }

    uint16_t *sp = (uint16_t *)cp;

    int out_end = (out_sz&~(NX-1));
    const uint32_t mask = (1u << TF_SHIFT)-1;

    __m128i maskv  = _mm_set1_epi32(mask); // set mask in all lanes
    LOAD128(Rv, R);

    for (i=0; i < out_end; i+=NX) {
	//for (z = 0; z < NX; z++)
	//  m[z] = R[z] & mask;
	__m128i masked1 = _mm_and_si128(Rv1, maskv);
	__m128i masked2 = _mm_and_si128(Rv2, maskv);
	__m128i masked3 = _mm_and_si128(Rv3, maskv);
	__m128i masked4 = _mm_and_si128(Rv4, maskv);

	//  S[z] = s3[m[z]];
	__m128i Sv1 = _mm_i32gather_epi32x((int *)s3, masked1, sizeof(*s3));
	__m128i Sv2 = _mm_i32gather_epi32x((int *)s3, masked2, sizeof(*s3));
	__m128i Sv3 = _mm_i32gather_epi32x((int *)s3, masked3, sizeof(*s3));
	__m128i Sv4 = _mm_i32gather_epi32x((int *)s3, masked4, sizeof(*s3));

	//  f[z] = S[z]>>(TF_SHIFT+8);
	__m128i fv1 = _mm_srli_epi32(Sv1, TF_SHIFT+8);
	__m128i fv2 = _mm_srli_epi32(Sv2, TF_SHIFT+8);
	__m128i fv3 = _mm_srli_epi32(Sv3, TF_SHIFT+8);
	__m128i fv4 = _mm_srli_epi32(Sv4, TF_SHIFT+8);

	//  b[z] = (S[z]>>8) & mask;
	__m128i bv1 = _mm_and_si128(_mm_srli_epi32(Sv1, 8), maskv);
	__m128i bv2 = _mm_and_si128(_mm_srli_epi32(Sv2, 8), maskv);
	__m128i bv3 = _mm_and_si128(_mm_srli_epi32(Sv3, 8), maskv);
	__m128i bv4 = _mm_and_si128(_mm_srli_epi32(Sv4, 8), maskv);

	//  s[z] = S[z] & 0xff;
	__m128i sv1 = _mm_and_si128(Sv1, _mm_set1_epi32(0xff));
	__m128i sv2 = _mm_and_si128(Sv2, _mm_set1_epi32(0xff));
	__m128i sv3 = _mm_and_si128(Sv3, _mm_set1_epi32(0xff));
	__m128i sv4 = _mm_and_si128(Sv4, _mm_set1_epi32(0xff));

	//  R[z] = f[z] * (R[z] >> TF_SHIFT) + b[z];
	Rv1 = _mm_add_epi32(
	          _mm_mullo_epi32(
		      _mm_srli_epi32(Rv1,TF_SHIFT), fv1), bv1);
	Rv2 = _mm_add_epi32(
		  _mm_mullo_epi32(
		      _mm_srli_epi32(Rv2,TF_SHIFT), fv2), bv2);
	Rv3 = _mm_add_epi32(
	          _mm_mullo_epi32(
		      _mm_srli_epi32(Rv3,TF_SHIFT), fv3), bv3);
	Rv4 = _mm_add_epi32(
		  _mm_mullo_epi32(
		      _mm_srli_epi32(Rv4,TF_SHIFT), fv4), bv4);

	// Tricky one:  out[i+z] = s[z];
	//             ---d---c ---b---a  sv1
	//             ---h---g ---f---e  sv2
	// packs_epi32 -h-g-f-e -d-c-b-a  sv1(2)
	// packs_epi16 ponmlkji hgfedcba  sv1(2) / sv3(4)
	sv1 = _mm_packus_epi32(sv1, sv2);
	sv3 = _mm_packus_epi32(sv3, sv4);
	sv1 = _mm_packus_epi16(sv1, sv3);

	// c =  R[z] < RANS_BYTE_L;
	// A little tricky as we only have signed comparisons.
	// See https://stackoverflow.com/questions/32945410/sse2-intrinsics-comparing-unsigned-integers

#define _mm_cmplt_epu32_imm(a,b) _mm_andnot_si128(_mm_cmpeq_epi32(_mm_max_epu32((a),_mm_set1_epi32(b)), (a)), _mm_set1_epi32(-1));

//#define _mm_cmplt_epu32_imm(a,b) _mm_cmpgt_epi32(_mm_set1_epi32((b)-0x80000000), _mm_xor_si128((a), _mm_set1_epi32(0x80000000)))

	__m128i renorm_mask1, renorm_mask2, renorm_mask3, renorm_mask4;
	renorm_mask1 = _mm_cmplt_epu32_imm(Rv1, RANS_BYTE_L);
	renorm_mask2 = _mm_cmplt_epu32_imm(Rv2, RANS_BYTE_L);
	renorm_mask3 = _mm_cmplt_epu32_imm(Rv3, RANS_BYTE_L);
	renorm_mask4 = _mm_cmplt_epu32_imm(Rv4, RANS_BYTE_L);

//#define P(A,B,C,D) ((A)+((B)<<2) + ((C)<<4) + ((D)<<6))
#define P(A,B,C,D) 				\
	{ A+0,A+1,A+2,A+3,			\
          B+0,B+1,B+2,B+3,			\
	  C+0,C+1,C+2,C+3,			\
	  D+0,D+1,D+2,D+3}
#ifdef _
#undef _
#endif
#define _ 0x80
	uint8_t pidx[16][16] = {
	    P(_,_,_,_),
	    P(0,_,_,_),
	    P(_,0,_,_),
	    P(0,4,_,_),

	    P(_,_,0,_),
	    P(0,_,4,_),
	    P(_,0,4,_),
	    P(0,4,8,_),

	    P(_,_,_,0),
	    P(0,_,_,4),
	    P(_,0,_,4),
	    P(0,4,_,8),

	    P(_,_,0,4),
	    P(0,_,4,8),
	    P(_,0,4,8),
	    P(0,4,8,12),
	};
#undef _

	// Shuffle the renorm values to correct lanes and incr sp pointer
	__m128i Vv1 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask1 = _mm_movemask_ps((__m128)renorm_mask1);
	Vv1 = _mm_shuffle_epi8(Vv1, _mm_load_si128((__m128i*)pidx[imask1]));
	sp += _mm_popcnt_u32(imask1);

	__m128i Vv2 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask2 = _mm_movemask_ps((__m128)renorm_mask2);
	sp += _mm_popcnt_u32(imask2);
	Vv2 = _mm_shuffle_epi8(Vv2, _mm_load_si128((__m128i*)pidx[imask2]));

	__m128i Vv3 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask3 = _mm_movemask_ps((__m128)renorm_mask3);
	Vv3 = _mm_shuffle_epi8(Vv3, _mm_load_si128((__m128i*)pidx[imask3]));
	sp += _mm_popcnt_u32(imask3);

	__m128i Vv4 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask4 = _mm_movemask_ps((__m128)renorm_mask4);
	sp += _mm_popcnt_u32(imask4);
	Vv4 = _mm_shuffle_epi8(Vv4, _mm_load_si128((__m128i*)pidx[imask4]));

	__m128i Yv1 = _mm_slli_epi32(Rv1, 16);
	__m128i Yv2 = _mm_slli_epi32(Rv2, 16);
	__m128i Yv3 = _mm_slli_epi32(Rv3, 16);
	__m128i Yv4 = _mm_slli_epi32(Rv4, 16);

	// y = (R[z] << 16) | V[z];
	Yv1 = _mm_or_si128(Yv1, Vv1);
	Yv2 = _mm_or_si128(Yv2, Vv2);
	Yv3 = _mm_or_si128(Yv3, Vv3);
	Yv4 = _mm_or_si128(Yv4, Vv4);

	// R[z] = c ? Y[z] : R[z];
	Rv1 = _mm_blendv_epi8(Rv1, Yv1, renorm_mask1);
	Rv2 = _mm_blendv_epi8(Rv2, Yv2, renorm_mask2);
	Rv3 = _mm_blendv_epi8(Rv3, Yv3, renorm_mask3);
	Rv4 = _mm_blendv_epi8(Rv4, Yv4, renorm_mask4);

	// ------------------------------------------------------------

	//  m[z] = R[z] & mask;
	__m128i masked5 = _mm_and_si128(Rv5, maskv);
	__m128i masked6 = _mm_and_si128(Rv6, maskv);
	__m128i masked7 = _mm_and_si128(Rv7, maskv);
	__m128i masked8 = _mm_and_si128(Rv8, maskv);

	//  S[z] = s3[m[z]];
	__m128i Sv5 = _mm_i32gather_epi32x((int *)s3, masked5, sizeof(*s3));
	__m128i Sv6 = _mm_i32gather_epi32x((int *)s3, masked6, sizeof(*s3));
	__m128i Sv7 = _mm_i32gather_epi32x((int *)s3, masked7, sizeof(*s3));
	__m128i Sv8 = _mm_i32gather_epi32x((int *)s3, masked8, sizeof(*s3));

	//  f[z] = S[z]>>(TF_SHIFT+8);
	__m128i fv5 = _mm_srli_epi32(Sv5, TF_SHIFT+8);
	__m128i fv6 = _mm_srli_epi32(Sv6, TF_SHIFT+8);
	__m128i fv7 = _mm_srli_epi32(Sv7, TF_SHIFT+8);
	__m128i fv8 = _mm_srli_epi32(Sv8, TF_SHIFT+8);

	//  b[z] = (S[z]>>8) & mask;
	__m128i bv5 = _mm_and_si128(_mm_srli_epi32(Sv5, 8), maskv);
	__m128i bv6 = _mm_and_si128(_mm_srli_epi32(Sv6, 8), maskv);
	__m128i bv7 = _mm_and_si128(_mm_srli_epi32(Sv7, 8), maskv);
	__m128i bv8 = _mm_and_si128(_mm_srli_epi32(Sv8, 8), maskv);

	//  s[z] = S[z] & 0xff;
	__m128i sv5 = _mm_and_si128(Sv5, _mm_set1_epi32(0xff));
	__m128i sv6 = _mm_and_si128(Sv6, _mm_set1_epi32(0xff));
	__m128i sv7 = _mm_and_si128(Sv7, _mm_set1_epi32(0xff));
	__m128i sv8 = _mm_and_si128(Sv8, _mm_set1_epi32(0xff));

	//  R[z] = f[z] * (R[z] >> TF_SHIFT) + b[z];
	Rv5 = _mm_add_epi32(
	          _mm_mullo_epi32(
		      _mm_srli_epi32(Rv5,TF_SHIFT), fv5), bv5);
	Rv6 = _mm_add_epi32(
		  _mm_mullo_epi32(
		      _mm_srli_epi32(Rv6,TF_SHIFT), fv6), bv6);
	Rv7 = _mm_add_epi32(
	          _mm_mullo_epi32(
		      _mm_srli_epi32(Rv7,TF_SHIFT), fv7), bv7);
	Rv8 = _mm_add_epi32(
		  _mm_mullo_epi32(
		      _mm_srli_epi32(Rv8,TF_SHIFT), fv8), bv8);

	// Tricky one:  out[i+z] = s[z];
	//             ---d---c ---b---a  sv1
	//             ---h---g ---f---e  sv2
	// packs_epi32 -h-g-f-e -d-c-b-a  sv1(2)
	// packs_epi16 ponmlkji hgfedcba  sv1(2) / sv3(4)
	sv5 = _mm_packus_epi32(sv5, sv6);
	sv7 = _mm_packus_epi32(sv7, sv8);
	sv5 = _mm_packus_epi16(sv5, sv7);

	// c =  R[z] < RANS_BYTE_L;
	__m128i renorm_mask5, renorm_mask6, renorm_mask7, renorm_mask8;
	renorm_mask5 = _mm_cmplt_epu32_imm(Rv5, RANS_BYTE_L);
	renorm_mask6 = _mm_cmplt_epu32_imm(Rv6, RANS_BYTE_L);
	renorm_mask7 = _mm_cmplt_epu32_imm(Rv7, RANS_BYTE_L);
	renorm_mask8 = _mm_cmplt_epu32_imm(Rv8, RANS_BYTE_L);
	
	// Shuffle the renorm values to correct lanes and incr sp pointer
	__m128i Vv5 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask5 = _mm_movemask_ps((__m128)renorm_mask5);
	Vv5 = _mm_shuffle_epi8(Vv5, _mm_load_si128((__m128i*)pidx[imask5]));
	sp += _mm_popcnt_u32(imask5);

	__m128i Vv6 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask6 = _mm_movemask_ps((__m128)renorm_mask6);
	sp += _mm_popcnt_u32(imask6);
	Vv6 = _mm_shuffle_epi8(Vv6, _mm_load_si128((__m128i*)pidx[imask6]));

	__m128i Vv7 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask7 = _mm_movemask_ps((__m128)renorm_mask7);
	Vv7 = _mm_shuffle_epi8(Vv7, _mm_load_si128((__m128i*)pidx[imask7]));
	sp += _mm_popcnt_u32(imask7);

	__m128i Vv8 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	unsigned int imask8 = _mm_movemask_ps((__m128)renorm_mask8);
	sp += _mm_popcnt_u32(imask8);
	Vv8 = _mm_shuffle_epi8(Vv8, _mm_load_si128((__m128i*)pidx[imask8]));

	__m128i Yv5 = _mm_slli_epi32(Rv5, 16);
	__m128i Yv6 = _mm_slli_epi32(Rv6, 16);
	__m128i Yv7 = _mm_slli_epi32(Rv7, 16);
	__m128i Yv8 = _mm_slli_epi32(Rv8, 16);

	// y = (R[z] << 16) | V[z];
	Yv5 = _mm_or_si128(Yv5, Vv5);
	Yv6 = _mm_or_si128(Yv6, Vv6);
	Yv7 = _mm_or_si128(Yv7, Vv7);
	Yv8 = _mm_or_si128(Yv8, Vv8);

	// R[z] = c ? Y[z] : R[z];
	Rv5 = _mm_blendv_epi8(Rv5, Yv5, renorm_mask5);
	Rv6 = _mm_blendv_epi8(Rv6, Yv6, renorm_mask6);
	Rv7 = _mm_blendv_epi8(Rv7, Yv7, renorm_mask7);
	Rv8 = _mm_blendv_epi8(Rv8, Yv8, renorm_mask8);

	// Maybe just a store128 instead?
	_mm_store_si128((__m128i *)&out[i+ 0], sv1);
	_mm_store_si128((__m128i *)&out[i+16], sv5);
//	*(uint64_t *)&out[i+ 0] = _mm_extract_epi64(sv1, 0);
//	*(uint64_t *)&out[i+ 8] = _mm_extract_epi64(sv1, 1);
//	*(uint64_t *)&out[i+16] = _mm_extract_epi64(sv5, 0);
//	*(uint64_t *)&out[i+24] = _mm_extract_epi64(sv5, 1);
    }

    STORE128(Rv, R);

    for (z = out_sz & (NX-1); z-- > 0; )
      out[out_end + z] = ssym[R[z] & mask];

    //fprintf(stderr, "    0 Decoded %d bytes\n", (int)(cp-in)); //c-size

    return out;

 err:
    free(out_free);
    return NULL;
}

#endif // HAVE_SSE4_1 and HAVE_SSSE3
