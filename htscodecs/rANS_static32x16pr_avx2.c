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

#ifdef HAVE_AVX2

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
#include "permute.h"

#define NX 32

#define LOAD1(a,b) __m256i a##1 = _mm256_load_si256((__m256i *)&b[0]);
#define LOAD2(a,b) __m256i a##2 = _mm256_load_si256((__m256i *)&b[8]);
#define LOAD3(a,b) __m256i a##3 = _mm256_load_si256((__m256i *)&b[16]);
#define LOAD4(a,b) __m256i a##4 = _mm256_load_si256((__m256i *)&b[24]);
#define LOAD(a,b) LOAD1(a,b);LOAD2(a,b);LOAD3(a,b);LOAD4(a,b)

#define STORE1(a,b) _mm256_store_si256((__m256i *)&b[0],  a##1);
#define STORE2(a,b) _mm256_store_si256((__m256i *)&b[8],  a##2);
#define STORE3(a,b) _mm256_store_si256((__m256i *)&b[16], a##3);
#define STORE4(a,b) _mm256_store_si256((__m256i *)&b[24], a##4);
#define STORE(a,b) STORE1(a,b);STORE2(a,b);STORE3(a,b);STORE4(a,b)

// _mm256__mul_epu32 is:
//  -b -d -f -h
//* -q -s -u -w
//= BQ DS FU HW where BQ=b*q etc
//
// We want
//  abcd efgh  (a)
// *pqrs tuvw  (b)
// =ABCD EFGH
//
// a    mul b      => BQ DS FU HW
// >>= 8           => -B QD SF UH
// &               => -B -D -F -H (1)
// a>>8 mul b>>8   => AP CR ET GV
// &               => A- C- E- G-
// | with (1)      => AB CD EF GH
#if 0
static __m256i _mm256_mulhi_epu32(__m256i a, __m256i b) {
    __m256i ab_lm = _mm256_mul_epu32(a, b);
    ab_lm = _mm256_srli_epi64(ab_lm, 32);
    a = _mm256_srli_epi64(a, 32);

    ab_lm = _mm256_and_si256(ab_lm, _mm256_set1_epi64x(0xffffffff));
    b = _mm256_srli_epi64(b, 32);

    __m256i ab_hm = _mm256_mul_epu32(a, b);
    ab_hm = _mm256_and_si256(ab_hm, _mm256_set1_epi64x((uint64_t)0xffffffff00000000)); 
    ab_hm = _mm256_or_si256(ab_hm, ab_lm);

    return ab_hm;
}
#else
static __m256i _mm256_mulhi_epu32(__m256i a, __m256i b) {
    // Multiply bottom 4 items and top 4 items together.
    __m256i ab_hm = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
    __m256i ab_lm = _mm256_srli_epi64(_mm256_mul_epu32(a, b), 32);

    // Shift to get hi 32-bit of each 64-bit product
    ab_hm = _mm256_and_si256(ab_hm, _mm256_set1_epi64x((uint64_t)0xffffffff00000000));

    return _mm256_or_si256(ab_lm, ab_hm);
}
#endif

#if 0
// Simulated gather.  This is sometimes faster as it can run on other ports.
static inline __m256i _mm256_i32gather_epi32x(int *b, __m256i idx, int size) {
    int c[8] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)c, idx);
    return _mm256_set_epi32(b[c[7]], b[c[6]], b[c[5]], b[c[4]], b[c[3]], b[c[2]], b[c[1]], b[c[0]]);
}
#else
#define _mm256_i32gather_epi32x _mm256_i32gather_epi32
#endif

unsigned char *rans_compress_O0_32x16_avx2(unsigned char *in, unsigned int in_size,
					   unsigned char *out, unsigned int *out_size) {
    unsigned char *cp, *out_end;
    RansEncSymbol syms[256];
    RansState ransN[NX] __attribute__((aligned(32)));
    uint8_t* ptr;
    uint32_t F[256+MAGIC] = {0};
    int i, j, tab_size = 0, rle, x, z;
    int bound = rans_compress_bound_4x16(in_size,0)-20; // -20 for order/size/meta
    uint32_t SB[256], SA[256], SD[256], SC[256];

    if (!out) {
	*out_size = bound;
	out = malloc(*out_size);
    }
    if (!out || bound > *out_size)
	return NULL;

    // If "out" isn't word aligned, tweak out_end/ptr to ensure it is.
    // We already added more round in bound to allow for this.
    if (((size_t)out)&1)
	bound--;
    ptr = out_end = out + bound;

    if (in_size == 0)
	goto empty;

    // Compute statistics
    hist8(in, in_size, F);

    // Normalise so frequences sum to power of 2
    uint32_t fsum = in_size;
    uint32_t max_val = round2(fsum);
    if (max_val > TOTFREQ)
	max_val = TOTFREQ;

    if (normalise_freq(F, fsum, max_val) < 0)
	return NULL;
    fsum=max_val;

    cp = out;
    cp += encode_freq(cp, F);
    tab_size = cp-out;
    //write(2, out+4, cp-(out+4));

    if (normalise_freq(F, fsum, TOTFREQ) < 0)
	return NULL;

    // Encode statistics.
    for (x = rle = j = 0; j < 256; j++) {
	if (F[j]) {
	    RansEncSymbolInit(&syms[j], x, F[j], TF_SHIFT);
	    x += F[j];
	}
    }

    for (z = 0; z < NX; z++)
      RansEncInit(&ransN[z]);

    z = i = in_size&(NX-1);
    while (z-- > 0)
      RansEncPutSymbol(&ransN[z], &ptr, &syms[in[in_size-(i-z)]]);

    // Build lookup tables for SIMD encoding
    for (i = 0; i < 256; i++) {
	SB[i] = syms[i].x_max;
	SA[i] = syms[i].rcp_freq;
	SD[i] = (syms[i].cmpl_freq<<0) | (syms[i].rcp_shift<<16);
	SC[i] = syms[i].bias;
    }

    uint16_t *ptr16 = (uint16_t *)ptr;

    LOAD(Rv, ransN);

    for (i=(in_size &~(NX-1)); i>0; i-=NX) {
	uint8_t *c = &in[i-NX];

// Set vs gather methods of loading data.
// Gather is faster, but can only schedule a few to run in parallel.
#define SET1(a,b) __m256i a##1 = _mm256_set_epi32(b[c[ 7]], b[c[ 6]], b[c[ 5]], b[c[ 4]], b[c[ 3]], b[c[ 2]], b[c[ 1]], b[c[ 0]])
#define SET2(a,b) __m256i a##2 = _mm256_set_epi32(b[c[15]], b[c[14]], b[c[13]], b[c[12]], b[c[11]], b[c[10]], b[c[ 9]], b[c[ 8]])
#define SET3(a,b) __m256i a##3 = _mm256_set_epi32(b[c[23]], b[c[22]], b[c[21]], b[c[20]], b[c[19]], b[c[18]], b[c[17]], b[c[16]])
#define SET4(a,b) __m256i a##4 = _mm256_set_epi32(b[c[31]], b[c[30]], b[c[29]], b[c[28]], b[c[27]], b[c[26]], b[c[25]], b[c[24]])
#define SET(a,b) SET1(a,b);SET2(a,b);SET3(a,b);SET4(a,b)

	// Renorm:
	// if (x > x_max) {*--ptr16 = x & 0xffff; x >>= 16;}
	SET(xmax, SB);
	__m256i cv1 = _mm256_cmpgt_epi32(Rv1, xmax1);
	__m256i cv2 = _mm256_cmpgt_epi32(Rv2, xmax2);
	__m256i cv3 = _mm256_cmpgt_epi32(Rv3, xmax3);
	__m256i cv4 = _mm256_cmpgt_epi32(Rv4, xmax4);

	// Store bottom 16-bits at ptr16
	unsigned int imask1 = _mm256_movemask_ps((__m256)cv1);
	unsigned int imask2 = _mm256_movemask_ps((__m256)cv2);
	unsigned int imask3 = _mm256_movemask_ps((__m256)cv3);
	unsigned int imask4 = _mm256_movemask_ps((__m256)cv4);

	__m256i idx1 = _mm256_load_si256((const __m256i*)permutec[imask1]);
	__m256i idx2 = _mm256_load_si256((const __m256i*)permutec[imask2]);
	__m256i idx3 = _mm256_load_si256((const __m256i*)permutec[imask3]);
	__m256i idx4 = _mm256_load_si256((const __m256i*)permutec[imask4]);

	// Permute; to gather together the rans states that need flushing
	__m256i V1 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv1, cv1), idx1);
	__m256i V2 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv2, cv2), idx2);
	__m256i V3 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv3, cv3), idx3);
	__m256i V4 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv4, cv4), idx4);
	
	// We only flush bottom 16 bits, to squash 32-bit states into 16 bit.
	V1 = _mm256_and_si256(V1, _mm256_set1_epi32(0xffff));
	V2 = _mm256_and_si256(V2, _mm256_set1_epi32(0xffff));
	V3 = _mm256_and_si256(V3, _mm256_set1_epi32(0xffff));
	V4 = _mm256_and_si256(V4, _mm256_set1_epi32(0xffff));
	__m256i V12 = _mm256_packus_epi32(V1, V2);
	__m256i V34 = _mm256_packus_epi32(V3, V4);

	// It's BAba order, want BbAa so shuffle.
	V12 = _mm256_permute4x64_epi64(V12, 0xd8);
	V34 = _mm256_permute4x64_epi64(V34, 0xd8);

	// Now we have bottom N 16-bit values in each V12/V34 to flush
	__m128i f =  _mm256_extractf128_si256(V34, 1);
	_mm_storeu_si128((__m128i *)(ptr16-8), f);
	ptr16 -= _mm_popcnt_u32(imask4);

	f =  _mm256_extractf128_si256(V34, 0);
	_mm_storeu_si128((__m128i *)(ptr16-8), f);
	ptr16 -= _mm_popcnt_u32(imask3);

	f =  _mm256_extractf128_si256(V12, 1);
	_mm_storeu_si128((__m128i *)(ptr16-8), f);
	ptr16 -= _mm_popcnt_u32(imask2);

	f =  _mm256_extractf128_si256(V12, 0);
	_mm_storeu_si128((__m128i *)(ptr16-8), f);
	ptr16 -= _mm_popcnt_u32(imask1);

	__m256i Rs;
	Rs = _mm256_srli_epi32(Rv1, 16); Rv1 = _mm256_blendv_epi8(Rv1, Rs, cv1);
	Rs = _mm256_srli_epi32(Rv2, 16); Rv2 = _mm256_blendv_epi8(Rv2, Rs, cv2);
	Rs = _mm256_srli_epi32(Rv3, 16); Rv3 = _mm256_blendv_epi8(Rv3, Rs, cv3);
	Rs = _mm256_srli_epi32(Rv4, 16); Rv4 = _mm256_blendv_epi8(Rv4, Rs, cv4);

	// Cannot trivially replace the multiply as mulhi_epu32 doesn't exist (only mullo).
	// However we can use _mm256_mul_epu32 twice to get 64bit results (half our lanes)
	// and shift/or to get the answer.
	//
	// (AVX512 allows us to hold it all in 64-bit lanes and use mullo_epi64
	// plus a shift.  KNC has mulhi_epi32, but not sure if this is available.)
	SET(rfv,   SA);

	rfv1 = _mm256_mulhi_epu32(Rv1, rfv1);
	rfv2 = _mm256_mulhi_epu32(Rv2, rfv2);
	rfv3 = _mm256_mulhi_epu32(Rv3, rfv3);
	rfv4 = _mm256_mulhi_epu32(Rv4, rfv4);

	SET(SDv,   SD);

	__m256i shiftv1 = _mm256_srli_epi32(SDv1, 16);
	__m256i shiftv2 = _mm256_srli_epi32(SDv2, 16);
	__m256i shiftv3 = _mm256_srli_epi32(SDv3, 16);
	__m256i shiftv4 = _mm256_srli_epi32(SDv4, 16);

	shiftv1 = _mm256_sub_epi32(shiftv1, _mm256_set1_epi32(32));
	shiftv2 = _mm256_sub_epi32(shiftv2, _mm256_set1_epi32(32));
	shiftv3 = _mm256_sub_epi32(shiftv3, _mm256_set1_epi32(32));
	shiftv4 = _mm256_sub_epi32(shiftv4, _mm256_set1_epi32(32));

	__m256i qv1 = _mm256_srlv_epi32(rfv1, shiftv1);
	__m256i qv2 = _mm256_srlv_epi32(rfv2, shiftv2);

	__m256i freqv1 = _mm256_and_si256(SDv1, _mm256_set1_epi32(0xffff));
	__m256i freqv2 = _mm256_and_si256(SDv2, _mm256_set1_epi32(0xffff));
	qv1 = _mm256_mullo_epi32(qv1, freqv1);
	qv2 = _mm256_mullo_epi32(qv2, freqv2);

	__m256i qv3 = _mm256_srlv_epi32(rfv3, shiftv3);
	__m256i qv4 = _mm256_srlv_epi32(rfv4, shiftv4);

	__m256i freqv3 = _mm256_and_si256(SDv3, _mm256_set1_epi32(0xffff));
	__m256i freqv4 = _mm256_and_si256(SDv4, _mm256_set1_epi32(0xffff));
	qv3 = _mm256_mullo_epi32(qv3, freqv3);
	qv4 = _mm256_mullo_epi32(qv4, freqv4);

	SET(biasv, SC);

	qv1 = _mm256_add_epi32(qv1, biasv1);
	qv2 = _mm256_add_epi32(qv2, biasv2);
	qv3 = _mm256_add_epi32(qv3, biasv3);
	qv4 = _mm256_add_epi32(qv4, biasv4);

	Rv1 = _mm256_add_epi32(Rv1, qv1);
	Rv2 = _mm256_add_epi32(Rv2, qv2);
	Rv3 = _mm256_add_epi32(Rv3, qv3);
	Rv4 = _mm256_add_epi32(Rv4, qv4);
    }

    STORE(Rv, ransN);

    ptr = (uint8_t *)ptr16;
    for (z = NX-1; z >= 0; z--)
      RansEncFlush(&ransN[z], &ptr);

 empty:
    // Finalise block size and return it
    *out_size = (out_end - ptr) + tab_size;

//    cp = out;
//    *cp++ = (in_size>> 0) & 0xff;
//    *cp++ = (in_size>> 8) & 0xff;
//    *cp++ = (in_size>>16) & 0xff;
//    *cp++ = (in_size>>24) & 0xff;

    memmove(out + tab_size, ptr, out_end-ptr);

    return out;
}

unsigned char *rans_uncompress_O0_32x16_avx2(unsigned char *in, unsigned int in_size,
					     unsigned char *out, unsigned int out_sz) {
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

    __m256i maskv  = _mm256_set1_epi32(mask); // set mask in all lanes
    LOAD(Rv, R);

    for (i=0; i < out_end; i+=NX) {
	//for (z = 0; z < NX; z++)
	//  m[z] = R[z] & mask;
	__m256i masked1 = _mm256_and_si256(Rv1, maskv);
	__m256i masked2 = _mm256_and_si256(Rv2, maskv);

	//  S[z] = s3[m[z]];
	__m256i Sv1 = _mm256_i32gather_epi32x((int *)s3, masked1, sizeof(*s3));
	__m256i Sv2 = _mm256_i32gather_epi32x((int *)s3, masked2, sizeof(*s3));

	//  f[z] = S[z]>>(TF_SHIFT+8);
	__m256i fv1 = _mm256_srli_epi32(Sv1, TF_SHIFT+8);
	__m256i fv2 = _mm256_srli_epi32(Sv2, TF_SHIFT+8);

	//  b[z] = (S[z]>>8) & mask;
	__m256i bv1 = _mm256_and_si256(_mm256_srli_epi32(Sv1, 8), maskv);
	__m256i bv2 = _mm256_and_si256(_mm256_srli_epi32(Sv2, 8), maskv);

	//  s[z] = S[z] & 0xff;
	__m256i sv1 = _mm256_and_si256(Sv1, _mm256_set1_epi32(0xff));
	__m256i sv2 = _mm256_and_si256(Sv2, _mm256_set1_epi32(0xff));

	//  R[z] = f[z] * (R[z] >> TF_SHIFT) + b[z];
	Rv1 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv1,TF_SHIFT),fv1),bv1);
	Rv2 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv2,TF_SHIFT),fv2),bv2);

	// Tricky one:  out[i+z] = s[z];
	//             ---h---g ---f---e  ---d---c ---b---a
	//             ---p---o ---n---m  ---l---k ---j---i
	// packs_epi32 -p-o-n-m -h-g-f-e  -l-k-j-i -d-c-b-a
	// permute4x64 -p-o-n-m -l-k-j-i  -h-g-f-e -d-c-b-a
	// packs_epi16 ponmlkji ponmlkji  hgfedcba hgfedcba
	sv1 = _mm256_packus_epi32(sv1, sv2);
	sv1 = _mm256_permute4x64_epi64(sv1, 0xd8);
	__m256i Vv1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	sv1 = _mm256_packus_epi16(sv1, sv1);

	// c =  R[z] < RANS_BYTE_L;
	__m256i renorm_mask1 = _mm256_xor_si256(Rv1, _mm256_set1_epi32(0x80000000));
	__m256i renorm_mask2 = _mm256_xor_si256(Rv2, _mm256_set1_epi32(0x80000000));
	renorm_mask1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask1);
	renorm_mask2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask2);
	
	// y = (R[z] << 16) | V[z];
	unsigned int imask1 = _mm256_movemask_ps((__m256)renorm_mask1);
	__m256i idx1 = _mm256_load_si256((const __m256i*)permute[imask1]);
	__m256i Yv1 = _mm256_slli_epi32(Rv1, 16);
	Vv1 = _mm256_permutevar8x32_epi32(Vv1, idx1);
	__m256i Yv2 = _mm256_slli_epi32(Rv2, 16);

	// Shuffle the renorm values to correct lanes and incr sp pointer
	unsigned int imask2 = _mm256_movemask_ps((__m256)renorm_mask2);
	sp += _mm_popcnt_u32(imask1);

	__m256i idx2 = _mm256_load_si256((const __m256i*)permute[imask2]);
	__m256i Vv2 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	sp += _mm_popcnt_u32(imask2);

	Yv1 = _mm256_or_si256(Yv1, Vv1);
	Vv2 = _mm256_permutevar8x32_epi32(Vv2, idx2);
	Yv2 = _mm256_or_si256(Yv2, Vv2);

	// R[z] = c ? Y[z] : R[z];
	Rv1 = _mm256_blendv_epi8(Rv1, Yv1, renorm_mask1);
	Rv2 = _mm256_blendv_epi8(Rv2, Yv2, renorm_mask2);

	// ------------------------------------------------------------

	//  m[z] = R[z] & mask;
	//  S[z] = s3[m[z]];
	__m256i masked3 = _mm256_and_si256(Rv3, maskv);
	__m256i Sv3 = _mm256_i32gather_epi32x((int *)s3, masked3, sizeof(*s3));

	*(uint64_t *)&out[i+0] = _mm256_extract_epi64(sv1, 0);
	*(uint64_t *)&out[i+8] = _mm256_extract_epi64(sv1, 2);

	__m256i masked4 = _mm256_and_si256(Rv4, maskv);
	__m256i Sv4 = _mm256_i32gather_epi32x((int *)s3, masked4, sizeof(*s3));

	//  f[z] = S[z]>>(TF_SHIFT+8);
	__m256i fv3 = _mm256_srli_epi32(Sv3, TF_SHIFT+8);
	__m256i fv4 = _mm256_srli_epi32(Sv4, TF_SHIFT+8);

	//  b[z] = (S[z]>>8) & mask;
	__m256i bv3 = _mm256_and_si256(_mm256_srli_epi32(Sv3, 8), maskv);
	__m256i bv4 = _mm256_and_si256(_mm256_srli_epi32(Sv4, 8), maskv);

	//  s[z] = S[z] & 0xff;
	__m256i sv3 = _mm256_and_si256(Sv3, _mm256_set1_epi32(0xff));
	__m256i sv4 = _mm256_and_si256(Sv4, _mm256_set1_epi32(0xff));

	//  R[z] = f[z] * (R[z] >> TF_SHIFT) + b[z];
	Rv3 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv3,TF_SHIFT),fv3),bv3);
	Rv4 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv4,TF_SHIFT),fv4),bv4);

	// Tricky one:  out[i+z] = s[z];
	//             ---h---g ---f---e  ---d---c ---b---a
	//             ---p---o ---n---m  ---l---k ---j---i
	// packs_epi32 -p-o-n-m -h-g-f-e  -l-k-j-i -d-c-b-a
	// permute4x64 -p-o-n-m -l-k-j-i  -h-g-f-e -d-c-b-a
	// packs_epi16 ponmlkji ponmlkji  hgfedcba hgfedcba
	sv3 = _mm256_packus_epi32(sv3, sv4);
	sv3 = _mm256_permute4x64_epi64(sv3, 0xd8);
	__m256i renorm_mask3 = _mm256_xor_si256(Rv3, _mm256_set1_epi32(0x80000000));
	__m256i renorm_mask4 = _mm256_xor_si256(Rv4, _mm256_set1_epi32(0x80000000));
	sv3 = _mm256_packus_epi16(sv3, sv3);
	// c =  R[z] < RANS_BYTE_L;

	renorm_mask3 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask3);
	renorm_mask4 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask4);
	
	*(uint64_t *)&out[i+16] = _mm256_extract_epi64(sv3, 0);
	*(uint64_t *)&out[i+24] = _mm256_extract_epi64(sv3, 2);

	// y = (R[z] << 16) | V[z];
	__m256i Vv3 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	__m256i Yv3 = _mm256_slli_epi32(Rv3, 16);
	unsigned int imask3 = _mm256_movemask_ps((__m256)renorm_mask3);
	__m256i idx3 = _mm256_load_si256((const __m256i*)permute[imask3]);

	// Shuffle the renorm values to correct lanes and incr sp pointer
	Vv3 = _mm256_permutevar8x32_epi32(Vv3, idx3);
	__m256i Yv4 = _mm256_slli_epi32(Rv4, 16);
	unsigned int imask4 = _mm256_movemask_ps((__m256)renorm_mask4);
	sp += _mm_popcnt_u32(imask3);

	__m256i idx4 = _mm256_load_si256((const __m256i*)permute[imask4]);
	__m256i Vv4 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));

	//Vv = _mm256_and_si256(Vv, renorm_mask);  (blend does the AND anyway)
	Yv3 = _mm256_or_si256(Yv3, Vv3);
	Vv4 = _mm256_permutevar8x32_epi32(Vv4, idx4);
	Yv4 = _mm256_or_si256(Yv4, Vv4);

	sp += _mm_popcnt_u32(imask4);

	// R[z] = c ? Y[z] : R[z];
	Rv3 = _mm256_blendv_epi8(Rv3, Yv3, renorm_mask3);
	Rv4 = _mm256_blendv_epi8(Rv4, Yv4, renorm_mask4);
    }

    STORE(Rv, R);
    //_mm256_store_si256((__m256i *)&R[0], Rv1);
    //_mm256_store_si256((__m256i *)&R[8], Rv2);
    //_mm256_store_si256((__m256i *)&R[16], Rv3);
    //_mm256_store_si256((__m256i *)&R[24], Rv4);

//#pragma omp simd
//	for (z = 0; z < NX; z++) {
//	  uint32_t m = R[z] & mask;
//	  R[z] = sfreq[m] * (R[z] >> TF_SHIFT) + sbase[m];
//	  out[i+z] = ssym[m];
//	  uint32_t c = R[z] < RANS_BYTE_L;  // NX16=>166MB/s
//	  uint32_t y = (R[z] << 16) | *spN[z];
//	  spN[z] += c ? 1 : 0;
//	  R[z]    = c ? y : R[z];
//
//	}
//    }

    for (z = out_sz & (NX-1); z-- > 0; )
      out[out_end + z] = ssym[R[z] & mask];

    //fprintf(stderr, "    0 Decoded %d bytes\n", (int)(cp-in)); //c-size

    return out;

 err:
    free(out_free);
    return NULL;
}


//-----------------------------------------------------------------------------

unsigned char *rans_compress_O1_32x16_avx2(unsigned char *in, unsigned int in_size,
					   unsigned char *out, unsigned int *out_size) {
    unsigned char *cp, *out_end, *op;
    unsigned int tab_size;
    RansEncSymbol syms[256][256];
    int bound = rans_compress_bound_4x16(in_size,1)-20, z;
    RansState ransN[NX] __attribute__((aligned(32)));

    if (in_size < NX) // force O0 instead
	return NULL;

    if (!out) {
	*out_size = bound;
	out = malloc(*out_size);
    }
    if (!out || bound > *out_size)
	return NULL;

    if (((size_t)out)&1)
	bound--;
    out_end = out + bound;

    uint32_t F[256][256] = {{0}}, T[256+MAGIC] = {0};
    int i, j;

    //memset(F, 0, 256*256*sizeof(int));
    //memset(T, 0, 256*sizeof(int));

    hist1_4(in, in_size, F, T);
    int isz4 = in_size/NX;
    for (z = 1; z < NX; z++)
	F[0][in[z*isz4]]++;
    T[0]+=NX-1;

    uint32_t F0[256+MAGIC] = {0};

    // Potential fix for the wrap-around bug in AVX2 O1 encoder with shift=12.
    // This occurs when we have one single symbol, giving freq=4096.
    // We fix it elsewhere for now by looking for the wrap-around.
    if (0) {
	int x = -1, y = -1;
	int n1, n2;
	for (x = 0; x < 256; x++) {
	    n1 = n2 = -1;
	    for (y = 0; y < 256; y++) {
		if (F[x][y])
		    n2 = n1, n1 = y;
	    }
	    if (n2!=-1 || n1 == -1)
		continue;

	    for (y = 0; y < 256; y++)
		if (!F[x][y])
		    break;
	    assert(y<256);
	    F[x][y]++;
	    F[0][y]++; T[y]++; F0[y]=1; 
	    F[0][x]++; T[x]++; F0[x]=1;
	}
    }


    op = cp = out;
    *cp++ = 0; // uncompressed header marker

    // Encode the order-0 symbols for use in the order-1 frequency tables
    //uint32_t F0[256+MAGIC] = {0};
    present8(in, in_size, F0);
    F0[0]=1;
    cp += encode_alphabet(cp, F0);

    // Decide between 10-bit and 12-bit freqs.
    // Fills out S[] to hold the new scaled maximum value.
    int S[256] = {0};
    int shift = compute_shift(F0, F, T, S);

    // Normalise so T[i] == TOTFREQ_O1
    for (i = 0; i < 256; i++) {
	unsigned int x;

	if (F0[i] == 0)
	    continue;

	int max_val = S[i];
	if (shift == TF_SHIFT_O1_FAST && max_val > TOTFREQ_O1_FAST)
	    max_val = TOTFREQ_O1_FAST;

//	if (max_val == TOTFREQ_O1_FAST)
//	    max_val--;

	if (normalise_freq(F[i], T[i], max_val) < 0)
	    return NULL;
	T[i]=max_val;

	cp += encode_freq_d(cp, F0, F[i]);

//	fprintf(stderr, "Normalise shift T[%d]=%d shift=%d\n", i, T[i], shift);
	normalise_freq_shift(F[i], T[i], 1<<shift); T[i]=1<<shift;

	uint32_t *F_i_ = F[i];
	for (x = j = 0; j < 256; j++) {
//	    fprintf(stderr, "x=%d F[%d][%d]=%d shift=%d\n", x, i, j, F_i_[j], shift);
	    RansEncSymbolInit(&syms[i][j], x, F_i_[j], shift);
	    x += F_i_[j];
	}
    }

    *op = shift<<4;
    if (cp - op > 1000) {
	// try rans0 compression of header
	unsigned int u_freq_sz = cp-(op+1);
	unsigned int c_freq_sz;
	unsigned char *c_freq = rans_compress_O0_4x16(op+1, u_freq_sz, NULL, &c_freq_sz);
	if (c_freq && c_freq_sz + 6 < cp-op) {
	    *op++ |= 1; // compressed
	    op += var_put_u32(op, NULL, u_freq_sz);
	    op += var_put_u32(op, NULL, c_freq_sz);
	    memcpy(op, c_freq, c_freq_sz);
	    cp = op+c_freq_sz;
	}
	free(c_freq);
    }

    //write(2, out+4, cp-(out+4));
    tab_size = cp - out;
    assert(tab_size < 257*257*3);

    for (z = 0; z < NX; z++)
      RansEncInit(&ransN[z]);

    uint8_t* ptr = out_end;

    int iN[NX];
    for (z = 0; z < NX; z++)
	iN[z] = (z+1)*isz4-2;

    unsigned char lN[NX];
    for (z = 0; z < NX; z++)
	lN[z] = in[iN[z]+1];

    // Deal with the remainder
    z = NX-1;
    lN[z] = in[in_size-1];
    for (iN[z] = in_size-2; iN[z] > NX*isz4-2; iN[z]--) {
	unsigned char c = in[iN[z]];
	RansEncPutSymbol(&ransN[z], &ptr, &syms[c][lN[z]]);
	lN[z] = c;
    }

    uint16_t *ptr16 = (uint16_t *)ptr;

    LOAD(Rv, ransN);

    for (; iN[0] >= 0; ) {
	uint32_t c[NX];

	// Gather all the symbol values together in adjacent arrays.
	// Better to just use raw set?
	RansEncSymbol_simd *sN[NX] __attribute__((aligned(32)));
	for (z = 0; z < NX; z++)
	    sN[z] = (RansEncSymbol_simd *)&syms[c[z] = in[iN[z]]][lN[z]];

#define SET1x(a,b,x) __m256i a##1 = _mm256_set_epi32(b[ 7]->x, b[ 6]->x, b[ 5]->x, b[ 4]->x, b[ 3]->x, b[ 2]->x, b[ 1]->x, b[ 0]->x)
#define SET2x(a,b,x) __m256i a##2 = _mm256_set_epi32(b[15]->x, b[14]->x, b[13]->x, b[12]->x, b[11]->x, b[10]->x, b[ 9]->x, b[ 8]->x)
#define SET3x(a,b,x) __m256i a##3 = _mm256_set_epi32(b[23]->x, b[22]->x, b[21]->x, b[20]->x, b[19]->x, b[18]->x, b[17]->x, b[16]->x)
#define SET4x(a,b,x) __m256i a##4 = _mm256_set_epi32(b[31]->x, b[30]->x, b[29]->x, b[28]->x, b[27]->x, b[26]->x, b[25]->x, b[24]->x)
#define SETx(a,b,x) SET1x(a,b,x);SET2x(a,b,x);SET3x(a,b,x);SET4x(a,b,x)

// As SET1x, but read x as a 32-bit value.
// This is used to load cmpl_freq and rcp_shift as a pair together
#if 0
#define SET1y(a,b,x) __m256i a##1 = _mm256_set_epi32(*(uint32_t *)&b[ 7]->x, *(uint32_t *)&b[ 6]->x, *(uint32_t *)&b[ 5]->x, *(uint32_t *)&b[ 4]->x, *(uint32_t *)&b[ 3]->x, *(uint32_t *)&b[ 2]->x, *(uint32_t *)&b[ 1]->x, *(uint32_t *)&b[ 0]->x)
#define SET2y(a,b,x) __m256i a##2 = _mm256_set_epi32(*(uint32_t *)&b[15]->x, *(uint32_t *)&b[14]->x, *(uint32_t *)&b[13]->x, *(uint32_t *)&b[12]->x, *(uint32_t *)&b[11]->x, *(uint32_t *)&b[10]->x, *(uint32_t *)&b[ 9]->x, *(uint32_t *)&b[ 8]->x)
#define SET3y(a,b,x) __m256i a##3 = _mm256_set_epi32(*(uint32_t *)&b[23]->x, *(uint32_t *)&b[22]->x, *(uint32_t *)&b[21]->x, *(uint32_t *)&b[20]->x, *(uint32_t *)&b[19]->x, *(uint32_t *)&b[18]->x, *(uint32_t *)&b[17]->x, *(uint32_t *)&b[16]->x)
#define SET4y(a,b,x) __m256i a##4 = _mm256_set_epi32(*(uint32_t *)&b[31]->x, *(uint32_t *)&b[30]->x, *(uint32_t *)&b[29]->x, *(uint32_t *)&b[28]->x, *(uint32_t *)&b[27]->x, *(uint32_t *)&b[26]->x, *(uint32_t *)&b[25]->x, *(uint32_t *)&b[24]->x)
#else
#define SET1y(a,b,x) __m256i a##1 = _mm256_set_epi32(b[ 7]->x, b[ 6]->x, b[ 5]->x, b[ 4]->x, b[ 3]->x, b[ 2]->x, b[ 1]->x, b[ 0]->x)
#define SET2y(a,b,x) __m256i a##2 = _mm256_set_epi32(b[15]->x, b[14]->x, b[13]->x, b[12]->x, b[11]->x, b[10]->x, b[ 9]->x, b[ 8]->x)
#define SET3y(a,b,x) __m256i a##3 = _mm256_set_epi32(b[23]->x, b[22]->x, b[21]->x, b[20]->x, b[19]->x, b[18]->x, b[17]->x, b[16]->x)
#define SET4y(a,b,x) __m256i a##4 = _mm256_set_epi32(b[31]->x, b[30]->x, b[29]->x, b[28]->x, b[27]->x, b[26]->x, b[25]->x, b[24]->x)
#endif
#define SETy(a,b,y) SET1y(a,b,y);SET2y(a,b,y);SET3y(a,b,y);SET4y(a,b,y)


	// ------------------------------------------------------------
	//	for (z = NX-1; z >= 0; z--) {
	//	    if (ransN[z] >= x_max[z]) {
	//		*--ptr16 = ransN[z] & 0xffff;
	//		ransN[z] >>= 16;
	//	    }
	//	}
	//LOAD(xmax,x_max);
	SETx(xmax, sN, x_max);
        __m256i cv1 = _mm256_cmpgt_epi32(Rv1, xmax1);
        __m256i cv2 = _mm256_cmpgt_epi32(Rv2, xmax2);
        __m256i cv3 = _mm256_cmpgt_epi32(Rv3, xmax3);
        __m256i cv4 = _mm256_cmpgt_epi32(Rv4, xmax4);

        // Store bottom 16-bits at ptr16                                                     
        //                                                                                   
        // for (z = NX-1; z >= 0; z--) {                                                     
        //     if (cond[z]) *--ptr16 = (uint16_t )(ransN[z] & 0xffff);                       
        // }                                                                                 
        unsigned int imask1 = _mm256_movemask_ps((__m256)cv1);
        unsigned int imask2 = _mm256_movemask_ps((__m256)cv2);
        unsigned int imask3 = _mm256_movemask_ps((__m256)cv3);
        unsigned int imask4 = _mm256_movemask_ps((__m256)cv4);

        __m256i idx1 = _mm256_load_si256((const __m256i*)permutec[imask1]);
        __m256i idx2 = _mm256_load_si256((const __m256i*)permutec[imask2]);
        __m256i idx3 = _mm256_load_si256((const __m256i*)permutec[imask3]);
        __m256i idx4 = _mm256_load_si256((const __m256i*)permutec[imask4]);

        // Permute; to gather together the rans states that need flushing                    
        __m256i V1 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv1, cv1), idx1);
        __m256i V2 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv2, cv2), idx2);
        __m256i V3 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv3, cv3), idx3);
        __m256i V4 = _mm256_permutevar8x32_epi32(_mm256_and_si256(Rv4, cv4), idx4);

        // We only flush bottom 16 bits, to squash 32-bit states into 16 bit.                
        V1 = _mm256_and_si256(V1, _mm256_set1_epi32(0xffff));
        V2 = _mm256_and_si256(V2, _mm256_set1_epi32(0xffff));
        V3 = _mm256_and_si256(V3, _mm256_set1_epi32(0xffff));
        V4 = _mm256_and_si256(V4, _mm256_set1_epi32(0xffff));
        __m256i V12 = _mm256_packus_epi32(V1, V2);
        __m256i V34 = _mm256_packus_epi32(V3, V4);

        // It's BAba order, want BbAa so shuffle.                                            
        V12 = _mm256_permute4x64_epi64(V12, 0xd8);
        V34 = _mm256_permute4x64_epi64(V34, 0xd8);
        // Now we have bottom N 16-bit values in each V12/V34 to flush                       
        __m128i f =  _mm256_extractf128_si256(V34, 1);
        _mm_storeu_si128((__m128i *)(ptr16-8), f);
        ptr16 -= _mm_popcnt_u32(imask4);

        f =  _mm256_extractf128_si256(V34, 0);
        _mm_storeu_si128((__m128i *)(ptr16-8), f);
        ptr16 -= _mm_popcnt_u32(imask3);

        f =  _mm256_extractf128_si256(V12, 1);
        _mm_storeu_si128((__m128i *)(ptr16-8), f);
        ptr16 -= _mm_popcnt_u32(imask2);

        f =  _mm256_extractf128_si256(V12, 0);
        _mm_storeu_si128((__m128i *)(ptr16-8), f);
        ptr16 -= _mm_popcnt_u32(imask1);

        __m256i Rs;
        Rs = _mm256_srli_epi32(Rv1, 16); Rv1 = _mm256_blendv_epi8(Rv1, Rs, cv1);
        Rs = _mm256_srli_epi32(Rv2, 16); Rv2 = _mm256_blendv_epi8(Rv2, Rs, cv2);
        Rs = _mm256_srli_epi32(Rv3, 16); Rv3 = _mm256_blendv_epi8(Rv3, Rs, cv3);
        Rs = _mm256_srli_epi32(Rv4, 16); Rv4 = _mm256_blendv_epi8(Rv4, Rs, cv4);

	// ------------------------------------------------------------
	// uint32_t q = (uint32_t) (((uint64_t)ransN[z] * rcp_freq[z]) >> rcp_shift[z]);
	// ransN[z] = ransN[z] + bias[z] + q * cmpl_freq[z];

        // Cannot trivially replace the multiply as mulhi_epu32 doesn't exist (only mullo).  
        // However we can use _mm256_mul_epu32 twice to get 64bit results (half our lanes)   
        // and shift/or to get the answer.                                                   
        //                                                                                   
        // (AVX512 allows us to hold it all in 64-bit lanes and use mullo_epi64              
        // plus a shift.  KNC has mulhi_epi32, but not sure if this is available.)
	SETx(rfv, sN, rcp_freq);

        rfv1 = _mm256_mulhi_epu32(Rv1, rfv1);
        rfv2 = _mm256_mulhi_epu32(Rv2, rfv2);
        rfv3 = _mm256_mulhi_epu32(Rv3, rfv3);
        rfv4 = _mm256_mulhi_epu32(Rv4, rfv4);

	//SETx(SDv, sN, SD); // where SD is cmp_freq | (rcp_shift<<16)
	SETy(SDv, sN, cmpl_freq);

        __m256i shiftv1 = _mm256_srli_epi32(SDv1, 16);
        __m256i shiftv2 = _mm256_srli_epi32(SDv2, 16);
        __m256i shiftv3 = _mm256_srli_epi32(SDv3, 16);
        __m256i shiftv4 = _mm256_srli_epi32(SDv4, 16);

        shiftv1 = _mm256_sub_epi32(shiftv1, _mm256_set1_epi32(32));
        shiftv2 = _mm256_sub_epi32(shiftv2, _mm256_set1_epi32(32));
        shiftv3 = _mm256_sub_epi32(shiftv3, _mm256_set1_epi32(32));
        shiftv4 = _mm256_sub_epi32(shiftv4, _mm256_set1_epi32(32));

        __m256i qv1 = _mm256_srlv_epi32(rfv1, shiftv1);
        __m256i qv2 = _mm256_srlv_epi32(rfv2, shiftv2);

        __m256i freqv1 = _mm256_and_si256(SDv1, _mm256_set1_epi32(0xffff));
        __m256i freqv2 = _mm256_and_si256(SDv2, _mm256_set1_epi32(0xffff));
        qv1 = _mm256_mullo_epi32(qv1, freqv1);
        qv2 = _mm256_mullo_epi32(qv2, freqv2);

        __m256i qv3 = _mm256_srlv_epi32(rfv3, shiftv3);
        __m256i qv4 = _mm256_srlv_epi32(rfv4, shiftv4);

        __m256i freqv3 = _mm256_and_si256(SDv3, _mm256_set1_epi32(0xffff));
        __m256i freqv4 = _mm256_and_si256(SDv4, _mm256_set1_epi32(0xffff));
        qv3 = _mm256_mullo_epi32(qv3, freqv3);
        qv4 = _mm256_mullo_epi32(qv4, freqv4);

	SETx(biasv, sN, bias);

        qv1 = _mm256_add_epi32(qv1, biasv1);
	qv2 = _mm256_add_epi32(qv2, biasv2);
        qv3 = _mm256_add_epi32(qv3, biasv3);
        qv4 = _mm256_add_epi32(qv4, biasv4);

	Rv1 = _mm256_add_epi32(Rv1, qv1);
        Rv2 = _mm256_add_epi32(Rv2, qv2);
        Rv3 = _mm256_add_epi32(Rv3, qv3);
        Rv4 = _mm256_add_epi32(Rv4, qv4);

	for (z = 0; z < NX; z++) {
	    //uint32_t q = (uint32_t) (((uint64_t)ransN[z] * rcp_freq[z]) >> rcp_shift[z]);
	    //ransN[z] = ransN[z] + bias[z] + q * cmpl_freq[z];
	    
	    lN[z] = c[z];
	    iN[z]--;
	}
    }

    STORE(Rv, ransN);

    ptr = (uint8_t *)ptr16;

    for (z = NX-1; z>=0; z--)
	RansEncPutSymbol(&ransN[z], &ptr, &syms[0][lN[z]]);

    for (z = NX-1; z>=0; z--)
	RansEncFlush(&ransN[z], &ptr);

    *out_size = (out_end - ptr) + tab_size;

    cp = out;
    memmove(out + tab_size, ptr, out_end-ptr);

    return out;
}

#ifndef NO_THREADS
/*
 * Thread local storage per thread in the pool.
 */
static pthread_once_t rans_once = PTHREAD_ONCE_INIT;
static pthread_key_t rans_key;

static void rans_tls_init(void) {
    pthread_key_create(&rans_key, free);
}
#endif

static inline void transpose_and_copy(uint8_t *out, int iN[32],
				      uint8_t t[32][32]) {
    int z;
//  for (z = 0; z < NX; z++) {
//      int k;
//      for (k = 0; k < 32; k++)
//  	out[iN[z]+k] = t[k][z];
//      iN[z] += 32;
//  }

    for (z = 0; z < NX; z+=4) {
	*(uint64_t *)&out[iN[z]] =
	    ((uint64_t)(t[0][z])<< 0) +
	    ((uint64_t)(t[1][z])<< 8) +
	    ((uint64_t)(t[2][z])<<16) +
	    ((uint64_t)(t[3][z])<<24) +
	    ((uint64_t)(t[4][z])<<32) +
	    ((uint64_t)(t[5][z])<<40) +
	    ((uint64_t)(t[6][z])<<48) +
	    ((uint64_t)(t[7][z])<<56);
	*(uint64_t *)&out[iN[z+1]] =
	    ((uint64_t)(t[0][z+1])<< 0) +
	    ((uint64_t)(t[1][z+1])<< 8) +
	    ((uint64_t)(t[2][z+1])<<16) +
	    ((uint64_t)(t[3][z+1])<<24) +
	    ((uint64_t)(t[4][z+1])<<32) +
	    ((uint64_t)(t[5][z+1])<<40) +
	    ((uint64_t)(t[6][z+1])<<48) +
	    ((uint64_t)(t[7][z+1])<<56);
	*(uint64_t *)&out[iN[z+2]] =
	    ((uint64_t)(t[0][z+2])<< 0) +
	    ((uint64_t)(t[1][z+2])<< 8) +
	    ((uint64_t)(t[2][z+2])<<16) +
	    ((uint64_t)(t[3][z+2])<<24) +
	    ((uint64_t)(t[4][z+2])<<32) +
	    ((uint64_t)(t[5][z+2])<<40) +
	    ((uint64_t)(t[6][z+2])<<48) +
	    ((uint64_t)(t[7][z+2])<<56);
	*(uint64_t *)&out[iN[z+3]] =
	    ((uint64_t)(t[0][z+3])<< 0) +
	    ((uint64_t)(t[1][z+3])<< 8) +
	    ((uint64_t)(t[2][z+3])<<16) +
	    ((uint64_t)(t[3][z+3])<<24) +
	    ((uint64_t)(t[4][z+3])<<32) +
	    ((uint64_t)(t[5][z+3])<<40) +
	    ((uint64_t)(t[6][z+3])<<48) +
	    ((uint64_t)(t[7][z+3])<<56);

	*(uint64_t *)&out[iN[z]+8] =
	    ((uint64_t)(t[8+0][z])<< 0) +
	    ((uint64_t)(t[8+1][z])<< 8) +
	    ((uint64_t)(t[8+2][z])<<16) +
	    ((uint64_t)(t[8+3][z])<<24) +
	    ((uint64_t)(t[8+4][z])<<32) +
	    ((uint64_t)(t[8+5][z])<<40) +
	    ((uint64_t)(t[8+6][z])<<48) +
	    ((uint64_t)(t[8+7][z])<<56);
	*(uint64_t *)&out[iN[z+1]+8] =
	    ((uint64_t)(t[8+0][z+1])<< 0) +
	    ((uint64_t)(t[8+1][z+1])<< 8) +
	    ((uint64_t)(t[8+2][z+1])<<16) +
	    ((uint64_t)(t[8+3][z+1])<<24) +
	    ((uint64_t)(t[8+4][z+1])<<32) +
	    ((uint64_t)(t[8+5][z+1])<<40) +
	    ((uint64_t)(t[8+6][z+1])<<48) +
	    ((uint64_t)(t[8+7][z+1])<<56);
	*(uint64_t *)&out[iN[z+2]+8] =
	    ((uint64_t)(t[8+0][z+2])<< 0) +
	    ((uint64_t)(t[8+1][z+2])<< 8) +
	    ((uint64_t)(t[8+2][z+2])<<16) +
	    ((uint64_t)(t[8+3][z+2])<<24) +
	    ((uint64_t)(t[8+4][z+2])<<32) +
	    ((uint64_t)(t[8+5][z+2])<<40) +
	    ((uint64_t)(t[8+6][z+2])<<48) +
	    ((uint64_t)(t[8+7][z+2])<<56);
	*(uint64_t *)&out[iN[z+3]+8] =
	    ((uint64_t)(t[8+0][z+3])<< 0) +
	    ((uint64_t)(t[8+1][z+3])<< 8) +
	    ((uint64_t)(t[8+2][z+3])<<16) +
	    ((uint64_t)(t[8+3][z+3])<<24) +
	    ((uint64_t)(t[8+4][z+3])<<32) +
	    ((uint64_t)(t[8+5][z+3])<<40) +
	    ((uint64_t)(t[8+6][z+3])<<48) +
	    ((uint64_t)(t[8+7][z+3])<<56);

	*(uint64_t *)&out[iN[z]+16] =
	    ((uint64_t)(t[16+0][z])<< 0) +
	    ((uint64_t)(t[16+1][z])<< 8) +
	    ((uint64_t)(t[16+2][z])<<16) +
	    ((uint64_t)(t[16+3][z])<<24) +
	    ((uint64_t)(t[16+4][z])<<32) +
	    ((uint64_t)(t[16+5][z])<<40) +
	    ((uint64_t)(t[16+6][z])<<48) +
	    ((uint64_t)(t[16+7][z])<<56);
	*(uint64_t *)&out[iN[z+1]+16] =
	    ((uint64_t)(t[16+0][z+1])<< 0) +
	    ((uint64_t)(t[16+1][z+1])<< 8) +
	    ((uint64_t)(t[16+2][z+1])<<16) +
	    ((uint64_t)(t[16+3][z+1])<<24) +
	    ((uint64_t)(t[16+4][z+1])<<32) +
	    ((uint64_t)(t[16+5][z+1])<<40) +
	    ((uint64_t)(t[16+6][z+1])<<48) +
	    ((uint64_t)(t[16+7][z+1])<<56);
	*(uint64_t *)&out[iN[z+2]+16] =
	    ((uint64_t)(t[16+0][z+2])<< 0) +
	    ((uint64_t)(t[16+1][z+2])<< 8) +
	    ((uint64_t)(t[16+2][z+2])<<16) +
	    ((uint64_t)(t[16+3][z+2])<<24) +
	    ((uint64_t)(t[16+4][z+2])<<32) +
	    ((uint64_t)(t[16+5][z+2])<<40) +
	    ((uint64_t)(t[16+6][z+2])<<48) +
	    ((uint64_t)(t[16+7][z+2])<<56);
	*(uint64_t *)&out[iN[z+3]+16] =
	    ((uint64_t)(t[16+0][z+3])<< 0) +
	    ((uint64_t)(t[16+1][z+3])<< 8) +
	    ((uint64_t)(t[16+2][z+3])<<16) +
	    ((uint64_t)(t[16+3][z+3])<<24) +
	    ((uint64_t)(t[16+4][z+3])<<32) +
	    ((uint64_t)(t[16+5][z+3])<<40) +
	    ((uint64_t)(t[16+6][z+3])<<48) +
	    ((uint64_t)(t[16+7][z+3])<<56);

	*(uint64_t *)&out[iN[z]+24] =
	    ((uint64_t)(t[24+0][z])<< 0) +
	    ((uint64_t)(t[24+1][z])<< 8) +
	    ((uint64_t)(t[24+2][z])<<16) +
	    ((uint64_t)(t[24+3][z])<<24) +
	    ((uint64_t)(t[24+4][z])<<32) +
	    ((uint64_t)(t[24+5][z])<<40) +
	    ((uint64_t)(t[24+6][z])<<48) +
	    ((uint64_t)(t[24+7][z])<<56);
	*(uint64_t *)&out[iN[z+1]+24] =
	    ((uint64_t)(t[24+0][z+1])<< 0) +
	    ((uint64_t)(t[24+1][z+1])<< 8) +
	    ((uint64_t)(t[24+2][z+1])<<16) +
	    ((uint64_t)(t[24+3][z+1])<<24) +
	    ((uint64_t)(t[24+4][z+1])<<32) +
	    ((uint64_t)(t[24+5][z+1])<<40) +
	    ((uint64_t)(t[24+6][z+1])<<48) +
	    ((uint64_t)(t[24+7][z+1])<<56);
	*(uint64_t *)&out[iN[z+2]+24] =
	    ((uint64_t)(t[24+0][z+2])<< 0) +
	    ((uint64_t)(t[24+1][z+2])<< 8) +
	    ((uint64_t)(t[24+2][z+2])<<16) +
	    ((uint64_t)(t[24+3][z+2])<<24) +
	    ((uint64_t)(t[24+4][z+2])<<32) +
	    ((uint64_t)(t[24+5][z+2])<<40) +
	    ((uint64_t)(t[24+6][z+2])<<48) +
	    ((uint64_t)(t[24+7][z+2])<<56);
	*(uint64_t *)&out[iN[z+3]+24] =
	    ((uint64_t)(t[24+0][z+3])<< 0) +
	    ((uint64_t)(t[24+1][z+3])<< 8) +
	    ((uint64_t)(t[24+2][z+3])<<16) +
	    ((uint64_t)(t[24+3][z+3])<<24) +
	    ((uint64_t)(t[24+4][z+3])<<32) +
	    ((uint64_t)(t[24+5][z+3])<<40) +
	    ((uint64_t)(t[24+6][z+3])<<48) +
	    ((uint64_t)(t[24+7][z+3])<<56);

	iN[z+0] += 32;
	iN[z+1] += 32;
	iN[z+2] += 32;
	iN[z+3] += 32;
    }
}

unsigned char *rans_uncompress_O1_32x16_avx2(unsigned char *in, unsigned int in_size,
					     unsigned char *out, unsigned int out_sz) {
    if (in_size < NX*4) // 4-states at least
	return NULL;

    if (out_sz >= INT_MAX)
	return NULL; // protect against some overflow cases

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    if (out_sz > 100000)
	return NULL;
#endif

    /* Load in the static tables */
    unsigned char *cp = in, *cp_end = in+in_size, *out_free = NULL;
    unsigned char *c_freq = NULL;
    int i, j = -999;
    unsigned int x;

#ifndef NO_THREADS
    pthread_once(&rans_once, rans_tls_init);
    uint8_t *s3_ = pthread_getspecific(rans_key);
    if (!s3_) {
	s3_ = malloc(256*TOTFREQ_O1*4);
	if (!s3_)
	    return NULL;
	pthread_setspecific(rans_key, s3_);
    }
    uint32_t (*s3)[TOTFREQ_O1] = (uint32_t (*)[TOTFREQ_O1])s3_;

#else
    uint32_t (*s3)[TOTFREQ_O1] = malloc(256*TOTFREQ_O1*4);
    if (!s3)
	return NULL;
#endif
    //uint32_t s3[256][TOTFREQ_O1] __attribute__((aligned(32)));
    uint32_t (*s3F)[TOTFREQ_O1_FAST] = (uint32_t (*)[TOTFREQ_O1_FAST])s3;

#ifdef VALIDATE
#define MAGIC2 179
    typedef struct {
	uint16_t f;
	uint16_t b;
    } fb_t;

    uint8_t *sfb_ = calloc(256*(TOTFREQ_O1+MAGIC2), sizeof(*sfb_));
    if (!sfb_)
	return NULL;
    fb_t fb[256][256];
    uint8_t *sfb[256];
    if ((*cp >> 4) == TF_SHIFT_O1) {
	for (i = 0; i < 256; i++)
	    sfb[i]=  sfb_ + i*(TOTFREQ_O1+MAGIC2);
    } else {
	for (i = 0; i < 256; i++)
	    sfb[i]=  sfb_ + i*(TOTFREQ_O1_FAST+MAGIC2);
    }
#endif

    if (!out)
	out_free = out = malloc(out_sz);

    if (!out)
	goto err;

    //fprintf(stderr, "out_sz=%d\n", out_sz);

    // compressed header? If so uncompress it
    unsigned char *tab_end = NULL;
    unsigned char *c_freq_end = cp_end;
    unsigned int shift = *cp >> 4;
    if (*cp++ & 1) {
	uint32_t u_freq_sz, c_freq_sz;
	cp += var_get_u32(cp, cp_end, &u_freq_sz);
	cp += var_get_u32(cp, cp_end, &c_freq_sz);
	if (c_freq_sz >= cp_end - cp - 16)
	    goto err;
	tab_end = cp + c_freq_sz;
	if (!(c_freq = rans_uncompress_O0_4x16(cp, c_freq_sz, NULL, u_freq_sz)))
	    goto err;
	cp = c_freq;
	c_freq_end = c_freq + u_freq_sz;
    }

    // Decode order-0 symbol list; avoids needing in order-1 tables
    uint32_t F0[256] = {0};
    int fsz = decode_alphabet(cp, c_freq_end, F0);
    if (!fsz)
	goto err;
    cp += fsz;

    if (cp >= c_freq_end)
	goto err;

    for (i = 0; i < 256; i++) {
	if (F0[i] == 0)
	    continue;

	uint32_t F[256] = {0}, T = 0;
	fsz = decode_freq_d(cp, c_freq_end, F0, F, &T);
	if (!fsz)
	    goto err;
	cp += fsz;

	if (!T) {
	    //fprintf(stderr, "No freq for F_%d\n", i);
	    continue;
	}

	normalise_freq_shift(F, T, 1<<shift);

	// Build symbols; fixme, do as part of decode, see the _d variant
	for (j = x = 0; j < 256; j++) {
	    if (F[j]) {
#ifdef VALIDATE
		memset(&sfb[i][x], j, F[j]);
		fb[i][j].f = F[j];
		fb[i][j].b = x;
#endif

		int y;
                for (y = 0; y < F[j]; y++) {
		    // s3 maps [last_sym][Rmask] to next_sym
		    if(shift == TF_SHIFT_O1)
			s3[i][y+x] = (((uint32_t)F[j])<<(shift+8)) | (y<<8) | j;
		    else
			// smaller matrix for better cache
			s3F[i][y+x] = (((uint32_t)F[j])<<(shift+8)) | (y<<8) | j;
		}

		x += F[j];
            }
	}
	if (x != (1<<shift))
	    // FIXME: if shift actually TF_SHIFT_O1 vs TF_SHIFT_O1_FAST
	    // or can it be smaller and permit upscaling (to keep
	    // freqs small).  If latter, check our uses of it for
	    // initialisation of s3 etc.

	    // we have O1 encoder writing (shift<<4) | do_comp for
	    // freq table.  Check encode / decode match.
	    // gdb -args ./tests/rans4x16pr -t -o 5 _
	    // in /nfs/users/nfs_j/jkb/work/samtools_master/htscodecs/build

	    goto err;
    }

    if (tab_end)
	cp = tab_end;
    free(c_freq);
    c_freq = NULL;

    if (cp+16 > cp_end)
	goto err;

    RansState R[NX] __attribute__((aligned(32)));
    uint8_t *ptr = cp, *ptr_end = in + in_size - 8;
    int z;
    for (z = 0; z < NX; z++) {
	RansDecInit(&R[z], &ptr);
	if (R[z] < RANS_BYTE_L)
	    goto err;
    }

    int isz4 = out_sz/NX;
    int iN[NX], lN[NX] __attribute__((aligned(32))) = {0};
    for (z = 0; z < NX; z++)
	iN[z] = z*isz4;

#ifdef VALIDATE
    RansState R_[NX];
    int i4[NX], l[NX] = {0};
    for (z = 0; z < NX; z++) {
	R_[z] = R[z];
	i4[z] = iN[z];
    }
#endif

    uint16_t *sp = (uint16_t *)ptr;
    const uint32_t mask = (1u << shift)-1;

    __m256i maskv  = _mm256_set1_epi32(mask);
    LOAD(Rv, R);
    LOAD(Lv, lN);

    union {
	unsigned char tbuf[32][32];
	uint64_t tbuf64[32][4];
    } u;
    unsigned int tidx = 0;

    if (shift == TF_SHIFT_O1) {
#ifdef VALIDATE
	const uint32_t mask = ((1u << TF_SHIFT_O1)-1);
#endif
	isz4 -= 64;
	for (; iN[0] < isz4; ) {
	    // m[z] = R[z] & mask;
	    __m256i masked1 = _mm256_and_si256(Rv1, maskv);
	    __m256i masked2 = _mm256_and_si256(Rv2, maskv);

	    //  S[z] = s3[lN[z]][m[z]];
	    Lv1 = _mm256_slli_epi32(Lv1, TF_SHIFT_O1);
	    masked1 = _mm256_add_epi32(masked1, Lv1);

	    Lv2 = _mm256_slli_epi32(Lv2, TF_SHIFT_O1);
	    masked2 = _mm256_add_epi32(masked2, Lv2);

	    __m256i masked3 = _mm256_and_si256(Rv3, maskv);
	    __m256i masked4 = _mm256_and_si256(Rv4, maskv);

	    Lv3 = _mm256_slli_epi32(Lv3, TF_SHIFT_O1);
	    masked3 = _mm256_add_epi32(masked3, Lv3);

	    Lv4 = _mm256_slli_epi32(Lv4, TF_SHIFT_O1);
	    masked4 = _mm256_add_epi32(masked4, Lv4);

	    __m256i Sv1 = _mm256_i32gather_epi32x((int *)&s3[0][0], masked1, sizeof(s3[0][0]));
	    __m256i Sv2 = _mm256_i32gather_epi32x((int *)&s3[0][0], masked2, sizeof(s3[0][0]));

	    //  f[z] = S[z]>>(TF_SHIFT_O1+8);
	    __m256i fv1 = _mm256_srli_epi32(Sv1, TF_SHIFT_O1+8);
	    __m256i fv2 = _mm256_srli_epi32(Sv2, TF_SHIFT_O1+8);

	    __m256i Sv3 = _mm256_i32gather_epi32x((int *)&s3[0][0], masked3, sizeof(s3[0][0]));
	    __m256i Sv4 = _mm256_i32gather_epi32x((int *)&s3[0][0], masked4, sizeof(s3[0][0]));

	    __m256i fv3 = _mm256_srli_epi32(Sv3, TF_SHIFT_O1+8);
	    __m256i fv4 = _mm256_srli_epi32(Sv4, TF_SHIFT_O1+8);

	    //  b[z] = (S[z]>>8) & mask;
	    __m256i bv1 = _mm256_and_si256(_mm256_srli_epi32(Sv1, 8), maskv);
	    __m256i bv2 = _mm256_and_si256(_mm256_srli_epi32(Sv2, 8), maskv);
	    __m256i bv3 = _mm256_and_si256(_mm256_srli_epi32(Sv3, 8), maskv);
	    __m256i bv4 = _mm256_and_si256(_mm256_srli_epi32(Sv4, 8), maskv);

	    //  s[z] = S[z] & 0xff;
	    __m256i sv1 = _mm256_and_si256(Sv1, _mm256_set1_epi32(0xff));
	    __m256i sv2 = _mm256_and_si256(Sv2, _mm256_set1_epi32(0xff));
	    __m256i sv3 = _mm256_and_si256(Sv3, _mm256_set1_epi32(0xff));
	    __m256i sv4 = _mm256_and_si256(Sv4, _mm256_set1_epi32(0xff));

	    if (1) {
		// A maximum frequency of 4096 doesn't fit in our s3 array.
		// as it's 12 bit + 12 bit + 8 bit.  It wraps around to zero.
		// (We don't have this issue for TOTFREQ_O1_FAST.)
		//
		// Solution 1 is to change to spec to forbid freq of 4096.
		// Easy hack is to add an extra symbol so it sums correctly.
		// => 572 MB/s on q40 (deskpro).
		//
		// Solution 2 implemented here is to look for the wrap around
		// and fix it.
		// => 556 MB/s on q40
		// cope with max freq of 4096.  Only 3% hit
		__m256i max_freq = _mm256_set1_epi32(TOTFREQ_O1);
		__m256i zero = _mm256_setzero_si256();
		__m256i cmp1 = _mm256_cmpeq_epi32(fv1, zero);
		fv1 = _mm256_blendv_epi8(fv1, max_freq, cmp1);
		__m256i cmp2 = _mm256_cmpeq_epi32(fv2, zero);
		fv2 = _mm256_blendv_epi8(fv2, max_freq, cmp2);
	    }

	    //  R[z] = f[z] * (R[z] >> TF_SHIFT_O1) + b[z];
	    Rv1 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv1,TF_SHIFT_O1),fv1),bv1);
	    Rv2 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv2,TF_SHIFT_O1),fv2),bv2);


	    //for (z = 0; z < NX; z++) lN[z] = c[z];
	    Lv1 = sv1;
	    Lv2 = sv2;

	    sv1 = _mm256_packus_epi32(sv1, sv2);
	    sv1 = _mm256_permute4x64_epi64(sv1, 0xd8);

	    // Start loading next batch of normalised states
	    __m256i Vv1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));

	    sv1 = _mm256_packus_epi16(sv1, sv1);

	    // out[iN[z]] = c[z];  // simulate scatter
	    // RansDecRenorm(&R[z], &ptr);	
	    __m256i renorm_mask1 = _mm256_xor_si256(Rv1, _mm256_set1_epi32(0x80000000));
	    __m256i renorm_mask2 = _mm256_xor_si256(Rv2, _mm256_set1_epi32(0x80000000));

	    renorm_mask1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask1);
	    renorm_mask2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask2);

	    unsigned int imask1 = _mm256_movemask_ps((__m256)renorm_mask1);
	    __m256i idx1 = _mm256_load_si256((const __m256i*)permute[imask1]);
	    __m256i Yv1 = _mm256_slli_epi32(Rv1, 16);
	    __m256i Yv2 = _mm256_slli_epi32(Rv2, 16);

	    unsigned int imask2 = _mm256_movemask_ps((__m256)renorm_mask2);
	    Vv1 = _mm256_permutevar8x32_epi32(Vv1, idx1);
	    sp += _mm_popcnt_u32(imask1);

	    __m256i idx2 = _mm256_load_si256((const __m256i*)permute[imask2]);
	    __m256i Vv2 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	    sp += _mm_popcnt_u32(imask2);
	    Vv2 = _mm256_permutevar8x32_epi32(Vv2, idx2);

	    //Vv = _mm256_and_si256(Vv, renorm_mask);  (blend does the AND anyway)
	    Yv1 = _mm256_or_si256(Yv1, Vv1);
	    Yv2 = _mm256_or_si256(Yv2, Vv2);

	    Rv1 = _mm256_blendv_epi8(Rv1, Yv1, renorm_mask1);
	    Rv2 = _mm256_blendv_epi8(Rv2, Yv2, renorm_mask2);

	    /////////////////////////////////////////////////////////////////////

	    // Start loading next batch of normalised states
	    __m256i Vv3 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));

	    if (1) {
		// cope with max freq of 4096
		__m256i max_freq = _mm256_set1_epi32(TOTFREQ_O1);
		__m256i zero = _mm256_setzero_si256();
		__m256i cmp3 = _mm256_cmpeq_epi32(fv3, zero);
		fv3 = _mm256_blendv_epi8(fv3, max_freq, cmp3);
		__m256i cmp4 = _mm256_cmpeq_epi32(fv4, zero);
		fv4 = _mm256_blendv_epi8(fv4, max_freq, cmp4);
	    }

	    //  R[z] = f[z] * (R[z] >> TF_SHIFT_O1) + b[z];
	    Rv3 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv3,TF_SHIFT_O1),fv3),bv3);
	    Rv4 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv4,TF_SHIFT_O1),fv4),bv4);

	    //for (z = 0; z < NX; z++) lN[z] = c[z];
	    Lv3 = sv3;
	    Lv4 = sv4;

	    // out[iN[z]] = c[z];  // simulate scatter
	    // RansDecRenorm(&R[z], &ptr);	
	    __m256i renorm_mask3 = _mm256_xor_si256(Rv3, _mm256_set1_epi32(0x80000000));
	    __m256i renorm_mask4 = _mm256_xor_si256(Rv4, _mm256_set1_epi32(0x80000000));

	    renorm_mask3 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask3);
	    renorm_mask4 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask4);

	    __m256i Yv3 = _mm256_slli_epi32(Rv3, 16);
	    __m256i Yv4 = _mm256_slli_epi32(Rv4, 16);

	    unsigned int imask3 = _mm256_movemask_ps((__m256)renorm_mask3);
	    unsigned int imask4 = _mm256_movemask_ps((__m256)renorm_mask4);
	    __m256i idx3 = _mm256_load_si256((const __m256i*)permute[imask3]);
	    sp += _mm_popcnt_u32(imask3);
	    Vv3 = _mm256_permutevar8x32_epi32(Vv3, idx3);

	    sv3 = _mm256_packus_epi32(sv3, sv4);
	    sv3 = _mm256_permute4x64_epi64(sv3, 0xd8);
	    sv3 = _mm256_packus_epi16(sv3, sv3);

	    u.tbuf64[tidx][0] = _mm256_extract_epi64(sv1, 0);
	    u.tbuf64[tidx][1] = _mm256_extract_epi64(sv1, 2);
	    u.tbuf64[tidx][2] = _mm256_extract_epi64(sv3, 0);
	    u.tbuf64[tidx][3] = _mm256_extract_epi64(sv3, 2);

	    iN[0]++;
	    if (++tidx == 32) {
		iN[0]-=32;

		transpose_and_copy(out, iN, u.tbuf);
		tidx = 0;
	    }

	    __m256i idx4 = _mm256_load_si256((const __m256i*)permute[imask4]);
	    __m256i Vv4 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));

	    //Vv = _mm256_and_si256(Vv, renorm_mask);  (blend does the AND anyway)
	    Yv3 = _mm256_or_si256(Yv3, Vv3);
	    Vv4 = _mm256_permutevar8x32_epi32(Vv4, idx4);
	    Yv4 = _mm256_or_si256(Yv4, Vv4);

	    sp += _mm_popcnt_u32(imask4);

	    Rv3 = _mm256_blendv_epi8(Rv3, Yv3, renorm_mask3);
	    Rv4 = _mm256_blendv_epi8(Rv4, Yv4, renorm_mask4);

#ifdef VALIDATE
	    STORE(Rv, R);
	    for (z = 0; z < NX; z+=4) {
		uint16_t m[4], c[4];
		c[0] = sfb[l[z+0]][m[0] = R_[z+0] & mask];
		c[1] = sfb[l[z+1]][m[1] = R_[z+1] & mask];
		c[2] = sfb[l[z+2]][m[2] = R_[z+2] & mask];
		c[3] = sfb[l[z+3]][m[3] = R_[z+3] & mask];
		
		R_[z+0] = fb[l[z+0]][c[0]].f * (R_[z+0]>>TF_SHIFT_O1);
		R_[z+0] += m[0] - fb[l[z+0]][c[0]].b;

		R_[z+1] = fb[l[z+1]][c[1]].f * (R_[z+1]>>TF_SHIFT_O1);
		R_[z+1] += m[1] - fb[l[z+1]][c[1]].b;

		R_[z+2] = fb[l[z+2]][c[2]].f * (R_[z+2]>>TF_SHIFT_O1);
		R_[z+2] += m[2] - fb[l[z+2]][c[2]].b;

		R_[z+3] = fb[l[z+3]][c[3]].f * (R_[z+3]>>TF_SHIFT_O1);
		R_[z+3] += m[3] - fb[l[z+3]][c[3]].b;

                i4[z+0]++; l[z+0] = c[0];
                i4[z+1]++; l[z+1] = c[1];
                i4[z+2]++; l[z+2] = c[2];
                i4[z+3]++; l[z+3] = c[3];

		//if (c[0] != out[iN[z+0]-1]) abort();
		//if (c[1] != out[iN[z+1]-1]) abort();
		//if (c[2] != out[iN[z+2]-1]) abort();
		//if (c[3] != out[iN[z+3]-1]) abort();

		if (ptr < ptr_end) {
		    RansDecRenorm(&R_[z+0], &ptr);
		    RansDecRenorm(&R_[z+1], &ptr);
		    RansDecRenorm(&R_[z+2], &ptr);
		    RansDecRenorm(&R_[z+3], &ptr);
		} else {
		    RansDecRenormSafe(&R_[z+0], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R_[z+1], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R_[z+2], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R_[z+3], &ptr, ptr_end+8);
		}
	    }

	    for (z = 0; z < NX; z++) {
		if (R[z] != R_[z]) {
		    fprintf(stderr, "iN[0] %d, z=%d\n", iN[0], z);
		    abort();
		}
	    }
	    // assert hits at loop 13503 with z==1.
	    // sp == ptr+2;  so we've moved on another item.
#endif
	}
	isz4 += 64;

	STORE(Rv, R);
	STORE(Lv, lN);
	ptr = (uint8_t *)sp;

	if (1) {
	    iN[0]-=tidx;
	    int T;
	    for (z = 0; z < NX; z++)
		for (T = 0; T < tidx; T++)
		    out[iN[z]++] = u.tbuf[T][z];
	}

	// Scalar version for close to the end of in[] array so we don't
	// do SIMD loads beyond the end of the buffer
	for (; iN[0] < isz4;) {
	    for (z = 0; z < NX; z++) {
		uint32_t m = R[z] & ((1u<<TF_SHIFT_O1)-1);
		uint32_t S = s3[lN[z]][m];
		unsigned char c = S & 0xff;
		out[iN[z]++] = c;
		uint32_t F = S>>(TF_SHIFT_O1+8);
		R[z] = (F?F:4096) * (R[z]>>TF_SHIFT_O1) +
		    ((S>>8) & ((1u<<TF_SHIFT_O1)-1));
		RansDecRenormSafe(&R[z], &ptr, ptr_end+8);
		lN[z] = c;
	    }
	}

	// Remainder
	z = NX-1;
	for (; iN[z] < out_sz; ) {
	    uint32_t m = R[z] & ((1u<<TF_SHIFT_O1)-1);
	    uint32_t S = s3[lN[z]][m];
	    unsigned char c = S & 0xff;
	    out[iN[z]++] = c;
	    uint32_t F = S>>(TF_SHIFT_O1+8);
	    R[z] = (F?F:4096) * (R[z]>>TF_SHIFT_O1) +
		((S>>8) & ((1u<<TF_SHIFT_O1)-1));
	    RansDecRenormSafe(&R[z], &ptr, ptr_end+8);
	    lN[z] = c;
	}
    } else {
	// TF_SHIFT_O1_FAST.  This is the most commonly used variant.

	// SIMD version ends decoding early as it reads at most 64 bytes
	// from input via 4 vectorised loads.
	isz4 -= 64;
	for (; iN[0] < isz4; ) {
	    // m[z] = R[z] & mask;
	    __m256i masked1 = _mm256_and_si256(Rv1, maskv);
	    __m256i masked2 = _mm256_and_si256(Rv2, maskv);

	    //  S[z] = s3[lN[z]][m[z]];
	    Lv1 = _mm256_slli_epi32(Lv1, TF_SHIFT_O1_FAST);
	    masked1 = _mm256_add_epi32(masked1, Lv1);

	    Lv2 = _mm256_slli_epi32(Lv2, TF_SHIFT_O1_FAST);
	    masked2 = _mm256_add_epi32(masked2, Lv2);

	    __m256i masked3 = _mm256_and_si256(Rv3, maskv);
	    __m256i masked4 = _mm256_and_si256(Rv4, maskv);

	    Lv3 = _mm256_slli_epi32(Lv3, TF_SHIFT_O1_FAST);
	    masked3 = _mm256_add_epi32(masked3, Lv3);

	    Lv4 = _mm256_slli_epi32(Lv4, TF_SHIFT_O1_FAST);
	    masked4 = _mm256_add_epi32(masked4, Lv4);

	    __m256i Sv1 = _mm256_i32gather_epi32x((int *)&s3F[0][0], masked1, sizeof(s3F[0][0]));
	    __m256i Sv2 = _mm256_i32gather_epi32x((int *)&s3F[0][0], masked2, sizeof(s3F[0][0]));

	    //  f[z] = S[z]>>(TF_SHIFT_O1+8);
	    __m256i fv1 = _mm256_srli_epi32(Sv1, TF_SHIFT_O1_FAST+8);
	    __m256i fv2 = _mm256_srli_epi32(Sv2, TF_SHIFT_O1_FAST+8);

	    __m256i Sv3 = _mm256_i32gather_epi32x((int *)&s3F[0][0], masked3, sizeof(s3F[0][0]));
	    __m256i Sv4 = _mm256_i32gather_epi32x((int *)&s3F[0][0], masked4, sizeof(s3F[0][0]));

	    __m256i fv3 = _mm256_srli_epi32(Sv3, TF_SHIFT_O1_FAST+8);
	    __m256i fv4 = _mm256_srli_epi32(Sv4, TF_SHIFT_O1_FAST+8);

	    //  b[z] = (S[z]>>8) & mask;
	    __m256i bv1 = _mm256_and_si256(_mm256_srli_epi32(Sv1, 8), maskv);
	    __m256i bv2 = _mm256_and_si256(_mm256_srli_epi32(Sv2, 8), maskv);
	    __m256i bv3 = _mm256_and_si256(_mm256_srli_epi32(Sv3, 8), maskv);
	    __m256i bv4 = _mm256_and_si256(_mm256_srli_epi32(Sv4, 8), maskv);

	    //  s[z] = S[z] & 0xff;
	    __m256i sv1 = _mm256_and_si256(Sv1, _mm256_set1_epi32(0xff));
	    __m256i sv2 = _mm256_and_si256(Sv2, _mm256_set1_epi32(0xff));
	    __m256i sv3 = _mm256_and_si256(Sv3, _mm256_set1_epi32(0xff));
	    __m256i sv4 = _mm256_and_si256(Sv4, _mm256_set1_epi32(0xff));

	    //  R[z] = f[z] * (R[z] >> TF_SHIFT_O1) + b[z];
	    Rv1 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv1,TF_SHIFT_O1_FAST),fv1),bv1);
	    Rv2 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv2,TF_SHIFT_O1_FAST),fv2),bv2);


	    //for (z = 0; z < NX; z++) lN[z] = c[z];
	    Lv1 = sv1;
	    Lv2 = sv2;

	    sv1 = _mm256_packus_epi32(sv1, sv2);
	    sv1 = _mm256_permute4x64_epi64(sv1, 0xd8);

	    // Start loading next batch of normalised states
	    __m256i Vv1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));

	    sv1 = _mm256_packus_epi16(sv1, sv1);

	    // out[iN[z]] = c[z];  // simulate scatter
	    // RansDecRenorm(&R[z], &ptr);	
	    __m256i renorm_mask1 = _mm256_xor_si256(Rv1, _mm256_set1_epi32(0x80000000));
	    __m256i renorm_mask2 = _mm256_xor_si256(Rv2, _mm256_set1_epi32(0x80000000));

	    renorm_mask1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask1);
	    renorm_mask2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask2);

	    unsigned int imask1 = _mm256_movemask_ps((__m256)renorm_mask1);
	    __m256i idx1 = _mm256_load_si256((const __m256i*)permute[imask1]);
	    __m256i Yv1 = _mm256_slli_epi32(Rv1, 16);
	    __m256i Yv2 = _mm256_slli_epi32(Rv2, 16);

	    unsigned int imask2 = _mm256_movemask_ps((__m256)renorm_mask2);
	    Vv1 = _mm256_permutevar8x32_epi32(Vv1, idx1);
	    sp += _mm_popcnt_u32(imask1);

	    __m256i idx2 = _mm256_load_si256((const __m256i*)permute[imask2]);
	    __m256i Vv2 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));
	    sp += _mm_popcnt_u32(imask2);
	    Vv2 = _mm256_permutevar8x32_epi32(Vv2, idx2);

	    //Vv = _mm256_and_si256(Vv, renorm_mask);  (blend does the AND anyway)
	    Yv1 = _mm256_or_si256(Yv1, Vv1);
	    Yv2 = _mm256_or_si256(Yv2, Vv2);

	    Rv1 = _mm256_blendv_epi8(Rv1, Yv1, renorm_mask1);
	    Rv2 = _mm256_blendv_epi8(Rv2, Yv2, renorm_mask2);

	    /////////////////////////////////////////////////////////////////////

	    // Start loading next batch of normalised states
	    __m256i Vv3 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));

	    //  R[z] = f[z] * (R[z] >> TF_SHIFT_O1) + b[z];
	    Rv3 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv3,TF_SHIFT_O1_FAST),fv3),bv3);
	    Rv4 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(Rv4,TF_SHIFT_O1_FAST),fv4),bv4);

	    //for (z = 0; z < NX; z++) lN[z] = c[z];
	    Lv3 = sv3;
	    Lv4 = sv4;

	    // out[iN[z]] = c[z];  // simulate scatter
	    // RansDecRenorm(&R[z], &ptr);	
	    __m256i renorm_mask3 = _mm256_xor_si256(Rv3, _mm256_set1_epi32(0x80000000));
	    __m256i renorm_mask4 = _mm256_xor_si256(Rv4, _mm256_set1_epi32(0x80000000));

	    renorm_mask3 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask3);
	    renorm_mask4 = _mm256_cmpgt_epi32(_mm256_set1_epi32(RANS_BYTE_L-0x80000000), renorm_mask4);

	    __m256i Yv3 = _mm256_slli_epi32(Rv3, 16);
	    __m256i Yv4 = _mm256_slli_epi32(Rv4, 16);

	    unsigned int imask3 = _mm256_movemask_ps((__m256)renorm_mask3);
	    unsigned int imask4 = _mm256_movemask_ps((__m256)renorm_mask4);
	    __m256i idx3 = _mm256_load_si256((const __m256i*)permute[imask3]);
	    sp += _mm_popcnt_u32(imask3);
	    Vv3 = _mm256_permutevar8x32_epi32(Vv3, idx3);

	    // sv3 sv4 are 32-bit ints with lowest bit being char
	    sv3 = _mm256_packus_epi32(sv3, sv4);       // 32 to 16; ABab
	    sv3 = _mm256_permute4x64_epi64(sv3, 0xd8); // shuffle;  AaBb
	    sv3 = _mm256_packus_epi16(sv3, sv3);       // 16 to 8

	    // Method 1
	    u.tbuf64[tidx][0] = _mm256_extract_epi64(sv1, 0);
	    u.tbuf64[tidx][1] = _mm256_extract_epi64(sv1, 2);
	    u.tbuf64[tidx][2] = _mm256_extract_epi64(sv3, 0);
	    u.tbuf64[tidx][3] = _mm256_extract_epi64(sv3, 2);

//	    // Method 2
//	    sv1 = _mm256_permute4x64_epi64(sv1, 8); // x x 10 00
//	    _mm_storeu_si128((__m128i *)&u.tbuf64[tidx][0],
//			     _mm256_extractf128_si256(sv1, 0));
//	    sv3 = _mm256_permute4x64_epi64(sv3, 8); // x x 10 00
//	    _mm_storeu_si128((__m128i *)&u.tbuf64[tidx][2],
//			     _mm256_extractf128_si256(sv3, 0));

//          // Method 3
//	    sv1 = _mm256_and_si256(sv1, _mm256_set_epi64x(0,-1,0,-1)); // AxBx
//	    sv3 = _mm256_and_si256(sv3, _mm256_set_epi64x(-1,0,-1,0)); // xCxD
//	    sv1 = _mm256_or_si256(sv1, sv3);                           // ACBD
//	    sv1 = _mm256_permute4x64_epi64(sv1, 0xD8); //rev 00 10 01 11; ABCD
//	    _mm256_storeu_si256((__m256i *)u.tbuf64[tidx], sv1);

	    iN[0]++;
	    if (++tidx == 32) {
		iN[0]-=32;

		// We have tidx[x][y] which we want to store in
		// memory in out[y][z] instead.  This is an unrolled
		// transposition.
		//
		// A straight memcpy (obviously wrong) decodes my test
		// data in around 1030MB/s vs 930MB/s for this transpose,
		// giving an idea of the time spent in this portion.
		transpose_and_copy(out, iN, u.tbuf);

		tidx = 0;
	    }

	    __m256i idx4 = _mm256_load_si256((const __m256i*)permute[imask4]);
	    __m256i Vv4 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)sp));

	    //Vv = _mm256_and_si256(Vv, renorm_mask);  (blend does the AND anyway)
	    Yv3 = _mm256_or_si256(Yv3, Vv3);
	    Vv4 = _mm256_permutevar8x32_epi32(Vv4, idx4);
	    Yv4 = _mm256_or_si256(Yv4, Vv4);

	    sp += _mm_popcnt_u32(imask4);

	    Rv3 = _mm256_blendv_epi8(Rv3, Yv3, renorm_mask3);
	    Rv4 = _mm256_blendv_epi8(Rv4, Yv4, renorm_mask4);
	}
	isz4 += 64;

	STORE(Rv, R);
	STORE(Lv, lN);
	ptr = (uint8_t *)sp;

	if (1) {
	    iN[0]-=tidx;
	    int T;
	    for (z = 0; z < NX; z++)
		for (T = 0; T < tidx; T++)
		    out[iN[z]++] = u.tbuf[T][z];
	}

	// Scalar version for close to the end of in[] array so we don't
	// do SIMD loads beyond the end of the buffer
	for (; iN[0] < isz4;) {
	    for (z = 0; z < NX; z++) {
		uint32_t m = R[z] & ((1u<<TF_SHIFT_O1_FAST)-1);
		uint32_t S = s3F[lN[z]][m];
		unsigned char c = S & 0xff;
		out[iN[z]++] = c;
		R[z] = (S>>(TF_SHIFT_O1_FAST+8)) * (R[z]>>TF_SHIFT_O1_FAST) +
		    ((S>>8) & ((1u<<TF_SHIFT_O1_FAST)-1));
		RansDecRenormSafe(&R[z], &ptr, ptr_end+8);
		lN[z] = c;
	    }
	}

	// Remainder
	z = NX-1;
	for (; iN[z] < out_sz; ) {
	    uint32_t m = R[z] & ((1u<<TF_SHIFT_O1_FAST)-1);
	    uint32_t S = s3F[lN[z]][m];
	    unsigned char c = S & 0xff;
	    out[iN[z]++] = c;
	    R[z] = (S>>(TF_SHIFT_O1_FAST+8)) * (R[z]>>TF_SHIFT_O1_FAST) +
		((S>>8) & ((1u<<TF_SHIFT_O1_FAST)-1));
	    RansDecRenormSafe(&R[z], &ptr, ptr_end+8);
	    lN[z] = c;
	}
    }

#ifdef NO_THREADS
    free(s3);
#endif
    return out;

 err:
#ifdef NO_THREADS
    free(s3);
#endif
    free(out_free);
    free(c_freq);

    return NULL;
}
#endif // HAVE_AVX2
