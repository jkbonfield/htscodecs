/*
 * Copyright (c) 2017-2020 Genome Research Ltd.
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

// As per standard rANS_static but using optional RLE or bit-packing
// techniques prior to entropy encoding.  This is a significant
// reduction in some data sets.

// top bits in order byte
#define X_PACK   0x80    // Pack 2,4,8 or infinite symbols into a byte.
#define X_RLE    0x40    // Run length encoding with runs & lits encoded separately
#define X_CAT    0x20    // Nop; for tiny segments where rANS overhead is too big
#define X_NOSZ   0x10    // Don't store the original size; used by STRIPE mode
#define X_STRIPE 0x08    // For N-byte integer data; rotate & encode N streams.
#define X_32     0x04    // 32-way unrolling instead of 4-way

// FIXME Can we get decoder to return the compressed sized read, avoiding
// us needing to store it?  Yes we can.  See c-size comments.  If we added all these
// together we could get rans_uncompress_to_4x16 to return the number of bytes
// consumed, avoiding the calling code from needed to explicitly stored the size.
// However the effect on name tokeniser is to save 0.1 to 0.2% so not worth it.

/*-------------------------------------------------------------------------- */
/*
 * Example wrapper to use the rans_byte.h functions included above.
 *
 * This demonstrates how to use, and unroll, an order-0 and order-1 frequency
 * model.
 */

#include "config.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#ifndef NO_THREADS
#include <pthread.h>
#endif

#include <x86intrin.h>

#include "rANS_word.h"
#include "rANS_static4x16.h"
#include "varint.h"
#include "pack.h"
#include "rle.h"
#include "utils.h"
#include "permute.h"

#define TF_SHIFT 12
#define TOTFREQ (1<<TF_SHIFT)

// 9-11 is considerably faster in the O1 variant due to reduced table size.
// We auto-tune between 10 and 12 though.  Anywhere from 9 to 14 are viable.
#ifndef TF_SHIFT_O1
#define TF_SHIFT_O1 12
#endif
#ifndef TF_SHIFT_O1_FAST
#define TF_SHIFT_O1_FAST 10
#endif
#define TOTFREQ_O1 (1<<TF_SHIFT_O1)
#define TOTFREQ_O1_FAST (1<<TF_SHIFT_O1_FAST)


/*-----------------------------------------------------------------------------
 * Memory to memory compression functions.
 *
 * These are original versions without any manual loop unrolling. They
 * are easier to understand, but can be up to 2x slower.
 */

// Rounds to next power of 2.
// credit to http://graphics.stanford.edu/~seander/bithacks.html
static uint32_t round2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static int normalise_freq(uint32_t *F, int size, uint32_t tot) {
    int m, M, j, loop = 0;
    uint64_t tr;
    if (!size)
	return 0;

 again:
    tr = ((uint64_t)tot<<31)/size + (1<<30)/size;

    for (size = m = M = j = 0; j < 256; j++) {
	if (!F[j])
	    continue;

	if (m < F[j])
	    m = F[j], M = j;

	if ((F[j] = (F[j]*tr)>>31) == 0)
	    F[j] = 1;
	size += F[j];
    }

    int adjust = tot - size;
    if (adjust > 0) {
	F[M] += adjust;
    } else if (adjust < 0) {
	if (F[M] > -adjust && (loop == 1 || F[M]/2 >= -adjust)) {
	    F[M] += adjust;
	} else {
	    if (loop < 1) {
		loop++;
		goto again;
	    }
	    adjust += F[M]-1;
	    F[M] = 1;
	    for (j = 0; adjust && j < 256; j++) {
		if (F[j] < 2) continue;

		int d = F[j] > -adjust;
		int m = d ? adjust : 1-F[j];
		F[j]   += m;
		adjust -= m;
	    }
	}
    }

    //printf("F[%d]=%d\n", M, F[M]);
    return F[M]>0 ? 0 : -1;
}

// A specialised version of normalise_freq_shift where the input size
// is already normalised to a power of 2, meaning we can just perform
// shifts instead of hard to define multiplications and adjustments.
static void normalise_freq_shift(uint32_t *F, uint32_t size,
				 uint32_t max_tot) {
    if (size == 0 || size == max_tot)
	return;

    int shift = 0, i;
    while (size < max_tot)
	size*=2, shift++;

    for (i = 0; i < 256; i++)
	F[i] <<= shift;
}

// symbols only
static int encode_alphabet(uint8_t *cp, uint32_t *F) {
    uint8_t *op = cp;
    int rle, j;

    for (rle = j = 0; j < 256; j++) {
	if (F[j]) {
	    // j
	    if (rle) {
		rle--;
	    } else {
		*cp++ = j;
		if (!rle && j && F[j-1])  {
		    for(rle=j+1; rle<256 && F[rle]; rle++)
			;
		    rle -= j+1;
		    *cp++ = rle;
		}
		//fprintf(stderr, "%d: %d %d\n", j, rle, N[j]);
	    }
	}
    }
    *cp++ = 0;
    
    return cp - op;
}

static int decode_alphabet(uint8_t *cp, uint8_t *cp_end, uint32_t *F) {
    if (cp == cp_end)
	return 0;

    uint8_t *op = cp;
    int rle = 0;
    int j = *cp++;
    if (cp+2 >= cp_end)
	goto carefully;

    do {
	F[j] = 1;
	if (!rle && j+1 == *cp) {
	    j = *cp++;
	    rle = *cp++;
	} else if (rle) {
	    rle--;
	    j++;
	    if (j > 255)
		return 0;
	} else {
	    j = *cp++;
	}
    } while(j && cp+2 < cp_end);

 carefully:
    if (j) {
	do {
	    F[j] = 1;
	    if(cp >= cp_end) return 0;
	    if (!rle && j+1 == *cp) {
		if (cp+1 >= cp_end) return 0;
		j = *cp++;
		rle = *cp++;
	    } else if (rle) {
		rle--;
		j++;
		if (j > 255)
		    return 0;
	    } else {
		if (cp >= cp_end) return 0;
		j = *cp++;
	    }
	} while(j && cp < cp_end);
    }

    return cp - op;
}

static int encode_freq(uint8_t *cp, uint32_t *F) {
    uint8_t *op = cp;
    int j;

    cp += encode_alphabet(cp, F);

    for (j = 0; j < 256; j++) {
	if (F[j])
	    cp += var_put_u32(cp, NULL, F[j]);
    }

    return cp - op;
}

static int decode_freq(uint8_t *cp, uint8_t *cp_end, uint32_t *F,
		       uint32_t *fsum) {
    if (cp == cp_end)
	return 0;

    uint8_t *op = cp;
    cp += decode_alphabet(cp, cp_end, F);

    int j, tot = 0;
    for (j = 0; j < 256; j++) {
	if (F[j]) {
	    cp += var_get_u32(cp, cp_end, (unsigned int *)&F[j]);
	    tot += F[j];
	}
    }

    *fsum = tot;
    return cp - op;
}


// Use the order-0 freqs in F0 to encode the order-1 stats in F.
// All symbols present in F are present in F0, but some in F0 will
// be empty in F.  Thus we run-length encode the 0 frequencies.
static int encode_freq_d(uint8_t *cp, uint32_t *F0, uint32_t *F) {
    uint8_t *op = cp;
    int j, dz;

    for (dz = j = 0; j < 256; j++) {
	if (F0[j]) {
	    if (F[j] != 0) {
		if (dz) {
		    // Replace dz zeros with zero + dz-1 run length
		    cp -= dz-1;
		    *cp++ = dz-1;
		}
		dz = 0;
		cp += var_put_u32(cp, NULL, F[j]);
	    } else {
		//fprintf(stderr, "2: j=%d F0[j]=%d, F[j]=%d, dz=%d\n", j, F0[j], F[j], dz);
		dz++;
		*cp++ = 0;
	    }
	} else {
	    assert(F[j] == 0);
	}
    }
    
    if (dz) {
	cp -= dz-1;
	*cp++ = dz-1;
    }

    return cp - op;
}

static int decode_freq_d(uint8_t *cp, uint8_t *cp_end, uint32_t *F0,
			 uint32_t *F, uint32_t *total) {
    if (cp == cp_end)
	return 0;

    uint8_t *op = cp;
    int j, dz, T = 0;

    for (j = dz = 0; j < 256 && cp < cp_end; j++) {
	//if (F0[j]) fprintf(stderr, "F0[%d]=%d\n", j, F0[j]);
	if (!F0[j])
	    continue;

	uint32_t f;
	if (dz) {
	    f = 0;
	    dz--;
	} else {
	    if (cp >= cp_end) return 0;
	    cp += var_get_u32(cp, cp_end, &f);
	    if (f == 0) {
		if (cp >= cp_end) return 0;
		dz = *cp++;
	    }
	}
	F[j] = f;
	T += f;
    }

    if (total) *total = T;
    return cp - op;
}

unsigned int rans_compress_bound_4x16(unsigned int size, int order) {
    int N = order>>8;
    if (!N) N=4;

    order &= 0xff;
    int sz = (order == 0
	? 1.05*size + 257*3 + 4
	: 1.05*size + 257*257*3 + 4 + 257*3+4) +
	((order & X_PACK) ? 1 : 0) +
	((order & X_RLE) ? 1 + 257*3+4: 0) + 20 +
	((order & X_32) ? 128 : 0) +
	((order & X_STRIPE) ? 1 + 5*N: 0);
    return sz + (sz&1) + 2; // make this even so buffers are word aligned
}

#ifdef NX

#if defined(USE_AVX2_ENC) || defined(USE_AVX2_DEC)
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
#endif

//#undef NX
//#define NX 4
//#undef USE_AVX2_ENC
//#undef USE_AVX2_DEC

#ifdef USE_AVX2_ENC
static
unsigned char *rans_compress_O0_4x16(unsigned char *in, unsigned int in_size,
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
#endif

#ifdef USE_AVX2_DEC
static
unsigned char *rans_uncompress_O0_4x16(unsigned char *in, unsigned int in_size,
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
#endif

#ifndef USE_AVX2_ENC
static
unsigned char *rans_compress_O0_4x16(unsigned char *in, unsigned int in_size,
				     unsigned char *out, unsigned int *out_size) {
    unsigned char *cp, *out_end;
    RansEncSymbol syms[256];
    RansState ransN[NX];
    uint8_t* ptr;
    uint32_t F[256+MAGIC] = {0};
    int i, j, tab_size = 0, rle, x, z;
    int bound = rans_compress_bound_4x16(in_size,0)-20; // -20 for order/size/meta

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

    for (i=(in_size &~(NX-1)); i>0; i-=NX) {
      for (z = NX-1; z >= 0; z-=4) {
	RansEncSymbol *s0 = &syms[in[i-(NX-z+0)]];
	RansEncSymbol *s1 = &syms[in[i-(NX-z+1)]];
	RansEncSymbol *s2 = &syms[in[i-(NX-z+2)]];
	RansEncSymbol *s3 = &syms[in[i-(NX-z+3)]];
	RansEncPutSymbol(&ransN[z-0], &ptr, s0);
	RansEncPutSymbol(&ransN[z-1], &ptr, s1);
	RansEncPutSymbol(&ransN[z-2], &ptr, s2);
	RansEncPutSymbol(&ransN[z-3], &ptr, s3);
	if (NX%8 == 0) {
	    z -= 4;
	    RansEncSymbol *s0 = &syms[in[i-(NX-z+0)]];
	    RansEncSymbol *s1 = &syms[in[i-(NX-z+1)]];
	    RansEncSymbol *s2 = &syms[in[i-(NX-z+2)]];
	    RansEncSymbol *s3 = &syms[in[i-(NX-z+3)]];
	    RansEncPutSymbol(&ransN[z-0], &ptr, s0);
	    RansEncPutSymbol(&ransN[z-1], &ptr, s1);
	    RansEncPutSymbol(&ransN[z-2], &ptr, s2);
	    RansEncPutSymbol(&ransN[z-3], &ptr, s3);
	}
      }
      if (z < -1) abort();
    }

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
#endif

#ifndef USE_AVX2_DEC
static
unsigned char *rans_uncompress_O0_4x16(unsigned char *in, unsigned int in_size,
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
    uint32_t s3[TOTFREQ]; // For TF_SHIFT <= 12

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
    RansState R[NX];
    for (z = 0; z < NX; z++) {
	RansDecInit(&R[z], &cp);
	if (R[z] < RANS_BYTE_L)
	    goto err;
    }

    int out_end = (out_sz&~(NX-1));
    const uint32_t mask = (1u << TF_SHIFT)-1;

    // assume NX is divisible by 4
    assert(NX%4==0);
    for (i=0; i < out_end; i+=NX) {
	for (z = 0; z < NX; z+=4) {
	    uint32_t S[4];
	    S[0] = s3[R[z+0] & mask];
	    S[1] = s3[R[z+1] & mask];
	    S[2] = s3[R[z+2] & mask];
	    S[3] = s3[R[z+3] & mask];

	    R[z+0] = (S[0]>>(TF_SHIFT+8)) * (R[z+0] >> TF_SHIFT) + ((S[0]>>8) & mask);
	    R[z+1] = (S[1]>>(TF_SHIFT+8)) * (R[z+1] >> TF_SHIFT) + ((S[1]>>8) & mask);
	    R[z+2] = (S[2]>>(TF_SHIFT+8)) * (R[z+2] >> TF_SHIFT) + ((S[2]>>8) & mask);
	    R[z+3] = (S[3]>>(TF_SHIFT+8)) * (R[z+3] >> TF_SHIFT) + ((S[3]>>8) & mask);

	    out[i+z+0] = S[0];
	    out[i+z+1] = S[1];
	    out[i+z+2] = S[2];
	    out[i+z+3] = S[3];

	    RansDecRenorm(&R[z+0], &cp);
	    RansDecRenorm(&R[z+1], &cp);
	    RansDecRenorm(&R[z+2], &cp);
	    RansDecRenorm(&R[z+3], &cp);

	    if (NX%8==0) {
		z += 4;
		S[0] = s3[R[z+0] & mask];
		S[1] = s3[R[z+1] & mask];
		S[2] = s3[R[z+2] & mask];
		S[3] = s3[R[z+3] & mask];

		R[z+0] = (S[0]>>(TF_SHIFT+8)) * (R[z+0] >> TF_SHIFT) + ((S[0]>>8) & mask);
		R[z+1] = (S[1]>>(TF_SHIFT+8)) * (R[z+1] >> TF_SHIFT) + ((S[1]>>8) & mask);
		R[z+2] = (S[2]>>(TF_SHIFT+8)) * (R[z+2] >> TF_SHIFT) + ((S[2]>>8) & mask);
		R[z+3] = (S[3]>>(TF_SHIFT+8)) * (R[z+3] >> TF_SHIFT) + ((S[3]>>8) & mask);

		out[i+z+0] = S[0];
		out[i+z+1] = S[1];
		out[i+z+2] = S[2];
		out[i+z+3] = S[3];

		RansDecRenorm(&R[z+0], &cp);
		RansDecRenorm(&R[z+1], &cp);
		RansDecRenorm(&R[z+2], &cp);
		RansDecRenorm(&R[z+3], &cp);
	    }
	}
    }

    for (z = out_sz & (NX-1); z-- > 0; )
      out[out_end + z] = ssym[R[z] & mask];

    //fprintf(stderr, "    0 Decoded %d bytes\n", (int)(cp-in)); //c-size

    return out;

 err:
    free(out_free);
    return NULL;
}
#endif

#else

// Compresses in_size bytes from 'in' to *out_size bytes in 'out'.
//
// NB: The output buffer does not hold the original size, so it is up to
// the caller to store this.
static
unsigned char *rans_compress_O0_4x16(unsigned char *in, unsigned int in_size,
				     unsigned char *out, unsigned int *out_size) {
    unsigned char *cp, *out_end;
    RansEncSymbol syms[256];
    RansState rans0;
    RansState rans2;
    RansState rans1;
    RansState rans3;
    uint8_t* ptr;
    uint32_t F[256+MAGIC] = {0};
    int i, j, tab_size = 0, rle, x;
    int bound = rans_compress_bound_4x16(in_size,0)-20; // -20 for order/size/meta

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

    RansEncInit(&rans0);
    RansEncInit(&rans1);
    RansEncInit(&rans2);
    RansEncInit(&rans3);

    switch (i=(in_size&3)) {
    case 3: RansEncPutSymbol(&rans2, &ptr, &syms[in[in_size-(i-2)]]);
    case 2: RansEncPutSymbol(&rans1, &ptr, &syms[in[in_size-(i-1)]]);
    case 1: RansEncPutSymbol(&rans0, &ptr, &syms[in[in_size-(i-0)]]);
    case 0:
	break;
    }
    for (i=(in_size &~3); i>0; i-=4) {
	RansEncSymbol *s3 = &syms[in[i-1]];
	RansEncSymbol *s2 = &syms[in[i-2]];
	RansEncSymbol *s1 = &syms[in[i-3]];
	RansEncSymbol *s0 = &syms[in[i-4]];

#if 1
	RansEncPutSymbol(&rans3, &ptr, s3);
	RansEncPutSymbol(&rans2, &ptr, s2);
	RansEncPutSymbol(&rans1, &ptr, s1);
	RansEncPutSymbol(&rans0, &ptr, s0);
#else
	// Slightly beter on gcc, much better on clang
	uint16_t *ptr16 = (uint16_t *)ptr;

	if (rans3 >= s3->x_max) *--ptr16 = (uint16_t)rans3, rans3 >>= 16;
	if (rans2 >= s2->x_max) *--ptr16 = (uint16_t)rans2, rans2 >>= 16;
	uint32_t q3 = (uint32_t) (((uint64_t)rans3 * s3->rcp_freq) >> s3->rcp_shift);
	uint32_t q2 = (uint32_t) (((uint64_t)rans2 * s2->rcp_freq) >> s2->rcp_shift);
	rans3 += s3->bias + q3 * s3->cmpl_freq;
	rans2 += s2->bias + q2 * s2->cmpl_freq;

	if (rans1 >= s1->x_max) *--ptr16 = (uint16_t)rans1, rans1 >>= 16;
	if (rans0 >= s0->x_max) *--ptr16 = (uint16_t)rans0, rans0 >>= 16;
	uint32_t q1 = (uint32_t) (((uint64_t)rans1 * s1->rcp_freq) >> s1->rcp_shift);
	uint32_t q0 = (uint32_t) (((uint64_t)rans0 * s0->rcp_freq) >> s0->rcp_shift);
	rans1 += s1->bias + q1 * s1->cmpl_freq;
	rans0 += s0->bias + q0 * s0->cmpl_freq;

	ptr = (uint8_t *)ptr16;
#endif
    }

    RansEncFlush(&rans3, &ptr);
    RansEncFlush(&rans2, &ptr);
    RansEncFlush(&rans1, &ptr);
    RansEncFlush(&rans0, &ptr);

 empty:
    // Finalise block size and return it
    *out_size = (out_end - ptr) + tab_size;

    memmove(out + tab_size, ptr, out_end-ptr);

    return out;
}

typedef struct {
    unsigned char R[TOTFREQ];
} ari_decoder;

static
unsigned char *rans_uncompress_O0_4x16(unsigned char *in, unsigned int in_size,
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
    uint16_t sfreq[TOTFREQ+32];
    uint16_t sbase[TOTFREQ+32]; // faster to use 32-bit on clang
    uint8_t  ssym [TOTFREQ+64]; // faster to use 16-bit on clang

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
		sfreq[y + x] = F[j];
		sbase[y + x] = y;
	    }
	    x += F[j];
	}
    }

    if (x != TOTFREQ)
	goto err;

    if (cp+16 > cp_end+8)
	goto err;

    RansState R[4];
    RansDecInit(&R[0], &cp); if (R[0] < RANS_BYTE_L) goto err;
    RansDecInit(&R[1], &cp); if (R[1] < RANS_BYTE_L) goto err;
    RansDecInit(&R[2], &cp); if (R[2] < RANS_BYTE_L) goto err;
    RansDecInit(&R[3], &cp); if (R[3] < RANS_BYTE_L) goto err;

// Simple version is comparable to below, but only with -O3
//
//    for (i = 0; cp < cp_end-8 && i < (out_sz&~7); i+=8) {
//        for(j=0; j<8;j++) {
//	    RansState m = RansDecGet(&R[j%4], TF_SHIFT);
//	    R[j%4] = sfreq[m] * (R[j%4] >> TF_SHIFT) + sbase[m];
//	    out[i+j] = ssym[m];
//	    RansDecRenorm(&R[j%4], &cp);
//        }
//    }

    for (i = 0; cp < cp_end-8 && i < (out_sz&~7); i+=8) {
	for (j = 0; j < 8; j+=4) {
	    RansState m0 = RansDecGet(&R[0], TF_SHIFT);
	    RansState m1 = RansDecGet(&R[1], TF_SHIFT);
	    R[0] = sfreq[m0] * (R[0] >> TF_SHIFT) + sbase[m0];
	    R[1] = sfreq[m1] * (R[1] >> TF_SHIFT) + sbase[m1];

	    RansDecRenorm(&R[0], &cp);
	    RansDecRenorm(&R[1], &cp);

	    out[i+j+0] = ssym[m0];
	    out[i+j+1] = ssym[m1];

	    RansState m3 = RansDecGet(&R[2], TF_SHIFT);
	    RansState m4 = RansDecGet(&R[3], TF_SHIFT);

	    R[2] = sfreq[m3] * (R[2] >> TF_SHIFT) + sbase[m3];
	    R[3] = sfreq[m4] * (R[3] >> TF_SHIFT) + sbase[m4];

	    out[i+j+2] = ssym[m3];
	    out[i+j+3] = ssym[m4];

	    RansDecRenorm(&R[2], &cp);
	    RansDecRenorm(&R[3], &cp);
	}
    }

    // remainder
    for (; i < out_sz; i++) {
        RansState m = RansDecGet(&R[i%4], TF_SHIFT);
	R[i%4] = sfreq[m] * (R[i%4] >> TF_SHIFT) + sbase[m];
	out[i] = ssym[m];
	RansDecRenormSafe(&R[i%4], &cp, cp_end+8);
    }

    //fprintf(stderr, "    0 Decoded %d bytes\n", (int)(cp-in)); //c-size

    return out;

 err:
    free(out_free);
    return NULL;
}
#endif

//-----------------------------------------------------------------------------

double fast_log(double a) {
  union { double d; long long x; } u = { a };
  return (u.x - 4606921278410026770) * 1.539095918623324e-16; /* 1 / 6497320848556798.0; */
}

// Compute the entropy of 12-bit vs 10-bit frequency tables.
// 10 bit means smaller memory footprint when decoding and
// more speed due to cache hits, but it *may* be a poor
// compression fit.
static int compute_shift(uint32_t *F0, uint32_t (*F)[256], uint32_t *T,
			 int *S) {
    int i, j;

    double e10 = 0, e12 = 0;
    int max_tot = 0;
    for (i = 0; i < 256; i++) {
	if (F0[i] == 0)
	    continue;
	int max_val = round2(T[i]);
	int ns = 0;
#define MAX(a,b) ((a)>(b)?(a):(b))

	// Number of samples that get their freq bumped to 1
	int sm10 = 0, sm12 = 0;
	for (j = 0; j < 256; j++) {
	    if (F[i][j] && max_val / F[i][j] > TOTFREQ_O1_FAST)
		sm10++;
	    if (F[i][j] && max_val / F[i][j] > TOTFREQ_O1)
		sm12++;
	}

	double l10 = log(TOTFREQ_O1_FAST + sm10);
	double l12 = log(TOTFREQ_O1      + sm12);

	for (j = 0; j < 256; j++) {
	    if (F[i][j]) {
		ns++;

		int x = (double)TOTFREQ_O1_FAST * F[i][j]/T[i];
		e10 -= F[i][j] * (fast_log(MAX(x,1)) - l10);

		x = (double)TOTFREQ_O1 * F[i][j]/T[i];
		e12 -= F[i][j] * (fast_log(MAX(x,1)) - l12);

		// Estimation of compressedf symbol freq table too.
		e10 += 4;
		e12 += 6;
	    }
	}

	// Order-1 frequencies often end up totalling under TOTFREQ.
	// In this case it's smaller to output the real frequencies
	// prior to normalisation and normalise after (with an extra
	// normalisation step needed in the decoder too).
	//
	// Thus we normalise to a power of 2 only, store those,
	// and renormalise later here (and in decoder) by bit-shift
	// to get to the fixed size.
	if (ns < 64 && max_val > 128) max_val /= 2;
	if (max_val > 1024)           max_val /= 2;
	if (max_val > TOTFREQ_O1)     max_val = TOTFREQ_O1;
	S[i] = max_val; // scale to max this
	if (max_tot < max_val)
	    max_tot = max_val;
    }
    int shift = e10/e12 < 1.01 || max_tot <= TOTFREQ_O1_FAST ? TF_SHIFT_O1_FAST : TF_SHIFT_O1;

//    fprintf(stderr, "e10/12 = %f %f %f, shift %d\n",
//    	    e10/log(256), e12/log(256), e10/e12, shift);

    return shift;
}

//#undef NX
//#define NX 32
//#define USE_AVX2_ENC
//#define USE_AVX2_DEC

#ifdef NX
static
unsigned char *rans_compress_O1_4x16(unsigned char *in, unsigned int in_size,
				     unsigned char *out, unsigned int *out_size) {
    unsigned char *cp, *out_end, *op;
    unsigned int tab_size;
    RansEncSymbol syms[256][256];
    int bound = rans_compress_bound_4x16(in_size,1)-20, z;
    RansState ransN[NX];

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

    // FIXME: Fix to prevent max freq.
    // Why not just cap at TOTFREQ_O1-1 instead?
    uint32_t F0[256+MAGIC] = {0};
    {
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
	    F[0][y]++;
	    F[0][x]++;
	    T[x]++;
	    T[y]++;
	    F0[x]=1; F0[y]=1;
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

	if (normalise_freq(F[i], T[i], max_val) < 0)
	    return NULL;
	T[i]=max_val;

	cp += encode_freq_d(cp, F0, F[i]);

	normalise_freq_shift(F[i], T[i], 1<<shift); T[i]=1<<shift;

	uint32_t *F_i_ = F[i];
	for (x = j = 0; j < 256; j++) {
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

    for (; iN[0] >= 0; ) {
	for (z = NX-1; z >= 0; z-=4) {
	    unsigned char c0;
	    unsigned char c1;
	    unsigned char c2;
	    unsigned char c3;

	    RansEncSymbol *s0 = &syms[c0 = in[iN[z-0]--]][lN[z-0]]; lN[z-0] = c0;
	    RansEncSymbol *s1 = &syms[c1 = in[iN[z-1]--]][lN[z-1]]; lN[z-1] = c1;
	    RansEncSymbol *s2 = &syms[c2 = in[iN[z-2]--]][lN[z-2]]; lN[z-2] = c2;
	    RansEncSymbol *s3 = &syms[c3 = in[iN[z-3]--]][lN[z-3]]; lN[z-3] = c3;

	    RansEncPutSymbol(&ransN[z-0], &ptr, s0);
	    RansEncPutSymbol(&ransN[z-1], &ptr, s1);
	    RansEncPutSymbol(&ransN[z-2], &ptr, s2);
	    RansEncPutSymbol(&ransN[z-3], &ptr, s3);
	}
    }

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
pthread_once_t rans_once = PTHREAD_ONCE_INIT;
pthread_key_t rans_key;

static void rans_tls_init(void) {
    pthread_key_create(&rans_key, free);
}
#endif

//#define MAGIC2 111
#define MAGIC2 179
//#define MAGIC2 0
typedef struct {
    uint16_t f;
    uint16_t b;
} fb_t;

#ifdef USE_AVX2_DEC
unsigned char *rans_uncompress_O1_4x16(unsigned char *in, unsigned int in_size,
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
    // FIXME: cope with MacOS and small stack. Use pthread_once trick.
    uint32_t s3[256][TOTFREQ_O1] __attribute__((aligned(32)));
    uint32_t (*s3F)[TOTFREQ_O1_FAST] = (uint32_t (*)[TOTFREQ_O1_FAST])s3;

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
		int y;
                for (y = 0; y < F[j]; y++)
		    // s3 maps [last_sym][Rmask] to next_sym
		    if(shift == TF_SHIFT_O1)
			s3[i][y+x] = (((uint32_t)F[j])<<(shift+8)) | (y<<8) | j;
		    else
			s3F[i][y+x] = (((uint32_t)F[j])<<(shift+8)) | (y<<8) | j;

		x += F[j];
            }
	}
	if (x != (1<<shift))
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
    int iN[NX], lN[NX] = {0};
    uint8_t c[NX] = {0};
    for (z = 0; z < NX; z++)
	iN[z] = z*isz4;

    uint16_t *sp = (uint16_t *)ptr;
    const uint32_t mask = (1u << shift)-1;

    __m256i maskv  = _mm256_set1_epi32(mask);
    LOAD(Rv, R);
    LOAD(Lv, lN);

    union {
	unsigned char tbuf[32][32];
	uint64_t tbuf64[32][4];
    } u;
    int tidx = 0;

    if (shift == TF_SHIFT_O1) {
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

	    //	// 12-bit fv may mean F==4096 wrapping to F==0 as no longer fitting.
	    //	uint32_t y[8];
	    //	_mm256_storeu_si256((__m256i *)y, fv1);
	    //	printf("< %04x %04x %04x %04x %04x %04x %04x %04x\n",
	    //	       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
	    //	for (z=0;z<8;z++)
	    //	    if (!y[z]) y[z]=TOTFREQ_O1;
	    //	printf("> %04x %04x %04x %04x %04x %04x %04x %04x\n",
	    //	       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
	    //	fv1 = _mm256_loadu_si256((__m256i *)y);
	    //
	    //	_mm256_storeu_si256((__m256i *)y, fv2);
	    //	for (z=0;z<8;z++)
	    //	    if (!y[z]) y[z]=TOTFREQ_O1;
	    //	fv2 = _mm256_loadu_si256((__m256i *)y);

	    //	__m256i max_freq = _mm256_set1_epi32(TOTFREQ_O1);
	    //	__m256i zero = _mm256_setzero_si256();
	    //	__m256i cmp1 = _mm256_cmpeq_epi32(fv1, zero);
	    //	fv1 = _mm256_blendv_epi8(fv1, max_freq, cmp1);
	    //	cmp1 = _mm256_cmpeq_epi32(fv2, zero);
	    //	fv2 = _mm256_blendv_epi8(fv2, max_freq, cmp1);

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

	    u.tbuf64[tidx][0] = _mm256_extract_epi64(sv1, 0);
	    u.tbuf64[tidx][1] = _mm256_extract_epi64(sv1, 2);

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

	    u.tbuf64[tidx][2] = _mm256_extract_epi64(sv3, 0);
	    u.tbuf64[tidx][3] = _mm256_extract_epi64(sv3, 2);

	    iN[0]++;
	    if (++tidx == 32) {
		iN[0]-=32;

		for (z = 0; z < NX; z++) {
		    // replace by gathers?
		    *(uint64_t *)&out[iN[z]] =
			((uint64_t)(u.tbuf[0][z])<< 0) + ((uint64_t)(u.tbuf[1][z])<< 8) +
			((uint64_t)(u.tbuf[2][z])<<16) + ((uint64_t)(u.tbuf[3][z])<<24) +
			((uint64_t)(u.tbuf[4][z])<<32) + ((uint64_t)(u.tbuf[5][z])<<40) +
			((uint64_t)(u.tbuf[6][z])<<48) + ((uint64_t)(u.tbuf[7][z])<<56);
		    *(uint64_t *)&out[iN[z]+8] =
			((uint64_t)(u.tbuf[8+0][z])<< 0) + ((uint64_t)(u.tbuf[8+1][z])<< 8) +
			((uint64_t)(u.tbuf[8+2][z])<<16) + ((uint64_t)(u.tbuf[8+3][z])<<24) +
			((uint64_t)(u.tbuf[8+4][z])<<32) + ((uint64_t)(u.tbuf[8+5][z])<<40) +
			((uint64_t)(u.tbuf[8+6][z])<<48) + ((uint64_t)(u.tbuf[8+7][z])<<56);
		    *(uint64_t *)&out[iN[z]+16] =
			((uint64_t)(u.tbuf[16+0][z])<< 0) + ((uint64_t)(u.tbuf[16+1][z])<< 8) +
			((uint64_t)(u.tbuf[16+2][z])<<16) + ((uint64_t)(u.tbuf[16+3][z])<<24) +
			((uint64_t)(u.tbuf[16+4][z])<<32) + ((uint64_t)(u.tbuf[16+5][z])<<40) +
			((uint64_t)(u.tbuf[16+6][z])<<48) + ((uint64_t)(u.tbuf[16+7][z])<<56);
		    *(uint64_t *)&out[iN[z]+24] =
			((uint64_t)(u.tbuf[24+0][z])<< 0) + ((uint64_t)(u.tbuf[24+1][z])<< 8) +
			((uint64_t)(u.tbuf[24+2][z])<<16) + ((uint64_t)(u.tbuf[24+3][z])<<24) +
			((uint64_t)(u.tbuf[24+4][z])<<32) + ((uint64_t)(u.tbuf[24+5][z])<<40) +
			((uint64_t)(u.tbuf[24+6][z])<<48) + ((uint64_t)(u.tbuf[24+7][z])<<56);
		    iN[z] += 32;
		}

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

	// Remainder
	z = NX-1;
	for (; iN[z] < out_sz; ) {
	    uint32_t m = R[z] & ((1u<<TF_SHIFT_O1)-1);
	    uint32_t S = s3[lN[z]][m];
	    unsigned char c = S & 0xff;
	    out[iN[z]++] = c;
	    R[z] = (S>>(TF_SHIFT_O1+8)) * (R[z]>>TF_SHIFT_O1) +
		((S>>8) & ((1u<<TF_SHIFT_O1)-1));
	    RansDecRenorm(&R[z], &ptr);
	    lN[z] = c;
	}
    } else {
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

	    u.tbuf64[tidx][0] = _mm256_extract_epi64(sv1, 0);
	    u.tbuf64[tidx][1] = _mm256_extract_epi64(sv1, 2);

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

	    sv3 = _mm256_packus_epi32(sv3, sv4);
	    sv3 = _mm256_permute4x64_epi64(sv3, 0xd8);
	    sv3 = _mm256_packus_epi16(sv3, sv3);

	    u.tbuf64[tidx][2] = _mm256_extract_epi64(sv3, 0);
	    u.tbuf64[tidx][3] = _mm256_extract_epi64(sv3, 2);

	    iN[0]++;
	    if (++tidx == 32) {
		iN[0]-=32;

		for (z = 0; z < NX; z++) {
		    // replace by gathers?
		    *(uint64_t *)&out[iN[z]] =
			((uint64_t)(u.tbuf[0][z])<< 0) + ((uint64_t)(u.tbuf[1][z])<< 8) +
			((uint64_t)(u.tbuf[2][z])<<16) + ((uint64_t)(u.tbuf[3][z])<<24) +
			((uint64_t)(u.tbuf[4][z])<<32) + ((uint64_t)(u.tbuf[5][z])<<40) +
			((uint64_t)(u.tbuf[6][z])<<48) + ((uint64_t)(u.tbuf[7][z])<<56);
		    *(uint64_t *)&out[iN[z]+8] =
			((uint64_t)(u.tbuf[8+0][z])<< 0) + ((uint64_t)(u.tbuf[8+1][z])<< 8) +
			((uint64_t)(u.tbuf[8+2][z])<<16) + ((uint64_t)(u.tbuf[8+3][z])<<24) +
			((uint64_t)(u.tbuf[8+4][z])<<32) + ((uint64_t)(u.tbuf[8+5][z])<<40) +
			((uint64_t)(u.tbuf[8+6][z])<<48) + ((uint64_t)(u.tbuf[8+7][z])<<56);
		    *(uint64_t *)&out[iN[z]+16] =
			((uint64_t)(u.tbuf[16+0][z])<< 0) + ((uint64_t)(u.tbuf[16+1][z])<< 8) +
			((uint64_t)(u.tbuf[16+2][z])<<16) + ((uint64_t)(u.tbuf[16+3][z])<<24) +
			((uint64_t)(u.tbuf[16+4][z])<<32) + ((uint64_t)(u.tbuf[16+5][z])<<40) +
			((uint64_t)(u.tbuf[16+6][z])<<48) + ((uint64_t)(u.tbuf[16+7][z])<<56);
		    *(uint64_t *)&out[iN[z]+24] =
			((uint64_t)(u.tbuf[24+0][z])<< 0) + ((uint64_t)(u.tbuf[24+1][z])<< 8) +
			((uint64_t)(u.tbuf[24+2][z])<<16) + ((uint64_t)(u.tbuf[24+3][z])<<24) +
			((uint64_t)(u.tbuf[24+4][z])<<32) + ((uint64_t)(u.tbuf[24+5][z])<<40) +
			((uint64_t)(u.tbuf[24+6][z])<<48) + ((uint64_t)(u.tbuf[24+7][z])<<56);
		    iN[z] += 32;
		}

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

	// Remainder
	z = NX-1;
	for (; iN[z] < out_sz; ) {
	    uint32_t m = R[z] & ((1u<<TF_SHIFT_O1_FAST)-1);
	    uint32_t S = s3F[lN[z]][m];
	    unsigned char c = S & 0xff;
	    out[iN[z]++] = c;
	    R[z] = (S>>(TF_SHIFT_O1_FAST+8)) * (R[z]>>TF_SHIFT_O1_FAST) +
		((S>>8) & ((1u<<TF_SHIFT_O1_FAST)-1));
	    RansDecRenorm(&R[z], &ptr);
	    lN[z] = c;
	}
    }

    return out;

 err:
    free(out_free);
    free(c_freq);

    return NULL;
}
#else
static
unsigned char *rans_uncompress_O1_4x16(unsigned char *in, unsigned int in_size,
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
    /*
     * The calloc below is expensive as it's a large structure.  We
     * could use malloc, but we're only initialising parts of the structure
     * that we need to, as dictated by the frequency table.  This is far
     * faster than initialising everything (ie malloc+memset => calloc).
     * Not initialising the data means malformed input with mismatching
     * frequency tables to actual data can lead to accessing of the
     * uninitialised sfb table and in turn potential leakage of the
     * uninitialised memory returned by malloc.  That could be anything at
     * all, including important encryption keys used within a server (for
     * example).
     *
     * However (I hope!) we don't care about leaking about the sfb symbol
     * frequencies previously computed by an earlier execution of *this*
     * code.  So calloc once and reuse is the fastest alternative.
     *
     * We do this through pthread local storage as we don't know if this
     * code is being executed in many threads simultaneously.
     */
    pthread_once(&rans_once, rans_tls_init);

    uint8_t *sfb_ = pthread_getspecific(rans_key);
    if (!sfb_) {
	sfb_ = calloc(256*(TOTFREQ_O1+MAGIC2), sizeof(*sfb_));
	pthread_setspecific(rans_key, sfb_);
    }
#else
    uint8_t *sfb_ = calloc(256*(TOTFREQ_O1+MAGIC2), sizeof(*sfb_));
#endif

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
		if (F[j] > (1<<shift) - x)
		    goto err;

		memset(&sfb[i][x], j, F[j]);
		fb[i][j].f = F[j];
		fb[i][j].b = x;
		x += F[j];
	    }
	}
	if (x != (1<<shift))
	    goto err;
    }

    if (tab_end)
	cp = tab_end;
    free(c_freq);
    c_freq = NULL;

    if (cp+16 > cp_end)
	goto err;

    RansState R[NX];
    uint8_t *ptr = cp, *ptr_end = in + in_size - 8;
    int z;
    for (z = 0; z < NX; z++) {
	RansDecInit(&R[z], &ptr);
	if (R[z] < RANS_BYTE_L)
	    goto err;
    }

    int isz4 = out_sz/NX;
    int i4[NX], l[NX] = {0};
    uint8_t c[NX] = {0};
    for (z = 0; z < NX; z++)
	i4[z] = z*isz4;

    // Around 15% faster to specialise for 10/12 than to have one
    // loop with shift as a variable.
    if (shift == TF_SHIFT_O1) {
	// TF_SHIFT_O1 = 12
	const uint32_t mask = ((1u << TF_SHIFT_O1)-1);
	for (; i4[0] < isz4;) {
	    for (z = 0; z < NX; z+=4) {
		uint16_t m[4], c[4];

		c[0] = sfb[l[z+0]][m[0] = R[z+0] & mask];
		c[1] = sfb[l[z+1]][m[1] = R[z+1] & mask];
		c[2] = sfb[l[z+2]][m[2] = R[z+2] & mask];
		c[3] = sfb[l[z+3]][m[3] = R[z+3] & mask];

		R[z+0] = fb[l[z+0]][c[0]].f * (R[z+0]>>TF_SHIFT_O1);
		R[z+0] += m[0] - fb[l[z+0]][c[0]].b;

		R[z+1] = fb[l[z+1]][c[1]].f * (R[z+1]>>TF_SHIFT_O1);
		R[z+1] += m[1] - fb[l[z+1]][c[1]].b;

		R[z+2] = fb[l[z+2]][c[2]].f * (R[z+2]>>TF_SHIFT_O1);
		R[z+2] += m[2] - fb[l[z+2]][c[2]].b;

		R[z+3] = fb[l[z+3]][c[3]].f * (R[z+3]>>TF_SHIFT_O1);
		R[z+3] += m[3] - fb[l[z+3]][c[3]].b;

		out[i4[z+0]++] = l[z+0] = c[0];
		out[i4[z+1]++] = l[z+1] = c[1];
		out[i4[z+2]++] = l[z+2] = c[2];
		out[i4[z+3]++] = l[z+3] = c[3];

		if (ptr < ptr_end) {
		    RansDecRenorm(&R[z+0], &ptr);
		    RansDecRenorm(&R[z+1], &ptr);
		    RansDecRenorm(&R[z+2], &ptr);
		    RansDecRenorm(&R[z+3], &ptr);
		} else {
		    RansDecRenormSafe(&R[z+0], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R[z+1], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R[z+2], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R[z+3], &ptr, ptr_end+8);
		}
	    }
	}

	// Remainder
	for (; i4[NX-1] < out_sz; i4[NX-1]++) {
	    uint32_t m = R[NX-1] & ((1u<<TF_SHIFT_O1)-1);
	    unsigned char c = sfb[l[NX-1]][m];
	    out[i4[NX-1]] = c;
	    R[NX-1] = fb[l[NX-1]][c].f * (R[NX-1]>>TF_SHIFT_O1) + m - fb[l[NX-1]][c].b;
	    RansDecRenormSafe(&R[NX-1], &ptr, ptr_end + 8);
	    l[NX-1] = c;
	}
    } else {
	// TF_SHIFT_O1 = 10
	const uint32_t mask = ((1u << TF_SHIFT_O1_FAST)-1);
	for (; i4[0] < isz4;) {
	    for (z = 0; z < NX; z+=4) {
		uint16_t m[4], c[4];

		c[0] = sfb[l[z+0]][m[0] = R[z+0] & mask];
		c[1] = sfb[l[z+1]][m[1] = R[z+1] & mask];
		c[2] = sfb[l[z+2]][m[2] = R[z+2] & mask];
		c[3] = sfb[l[z+3]][m[3] = R[z+3] & mask];

		R[z+0] = fb[l[z+0]][c[0]].f * (R[z+0]>>TF_SHIFT_O1_FAST);
		R[z+0] += m[0] - fb[l[z+0]][c[0]].b;

		R[z+1] = fb[l[z+1]][c[1]].f * (R[z+1]>>TF_SHIFT_O1_FAST);
		R[z+1] += m[1] - fb[l[z+1]][c[1]].b;

		R[z+2] = fb[l[z+2]][c[2]].f * (R[z+2]>>TF_SHIFT_O1_FAST);
		R[z+2] += m[2] - fb[l[z+2]][c[2]].b;

		R[z+3] = fb[l[z+3]][c[3]].f * (R[z+3]>>TF_SHIFT_O1_FAST);
		R[z+3] += m[3] - fb[l[z+3]][c[3]].b;

		out[i4[z+0]++] = l[z+0] = c[0];
		out[i4[z+1]++] = l[z+1] = c[1];
		out[i4[z+2]++] = l[z+2] = c[2];
		out[i4[z+3]++] = l[z+3] = c[3];

		if (ptr < ptr_end) {
		    RansDecRenorm(&R[z+0], &ptr);
		    RansDecRenorm(&R[z+1], &ptr);
		    RansDecRenorm(&R[z+2], &ptr);
		    RansDecRenorm(&R[z+3], &ptr);
		} else {
		    RansDecRenormSafe(&R[z+0], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R[z+1], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R[z+2], &ptr, ptr_end+8);
		    RansDecRenormSafe(&R[z+3], &ptr, ptr_end+8);
		}
	    }
	}

	// Remainder
	for (; i4[NX-1] < out_sz; i4[NX-1]++) {
	    uint32_t m = R[NX-1] & ((1u<<TF_SHIFT_O1_FAST)-1);
	    unsigned char c = sfb[l[NX-1]][m];
	    out[i4[NX-1]] = c;
	    R[NX-1] = fb[l[NX-1]][c].f * (R[NX-1]>>TF_SHIFT_O1_FAST) + m - fb[l[NX-1]][c].b;
	    RansDecRenormSafe(&R[NX-1], &ptr, ptr_end + 8);
	    l[NX-1] = c;
	}
    }
    //fprintf(stderr, "    1 Decoded %d bytes\n", (int)(ptr-in)); //c-size

#ifdef NO_THREADS
    free(sfb_);
#endif
    return out;

 err:
#ifdef NO_THREADS
    free(sfb_);
#endif
    free(out_free);
    free(c_freq);

    return NULL;
}
#endif

#else
static
unsigned char *rans_compress_O1_4x16(unsigned char *in, unsigned int in_size,
				     unsigned char *out, unsigned int *out_size) {
    unsigned char *cp, *out_end, *op;
    unsigned int tab_size;
    RansEncSymbol syms[256][256];
    int bound = rans_compress_bound_4x16(in_size,1)-20; // -20 for order/size/meta

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
    F[0][in[1*(in_size>>2)]]++;
    F[0][in[2*(in_size>>2)]]++;
    F[0][in[3*(in_size>>2)]]++;
    T[0]+=3;

    op = cp = out;
    *cp++ = 0; // uncompressed header marker

    // Encode the order-0 symbols for use in the order-1 frequency tables
    uint32_t F0[256+MAGIC] = {0};
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

	if (normalise_freq(F[i], T[i], max_val) < 0)
	    return NULL;
	T[i]=max_val;

	cp += encode_freq_d(cp, F0, F[i]);

	normalise_freq_shift(F[i], T[i], 1<<shift); T[i]=1<<shift;

	uint32_t *F_i_ = F[i];
	for (x = j = 0; j < 256; j++) {
	    RansEncSymbolInit(&syms[i][j], x, F_i_[j], shift);
	    x += F_i_[j];
	}

    }

    *op = shift<<4;
    if (cp - op > 1000 && 0) {
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

    RansState rans0, rans1, rans2, rans3;
    RansEncInit(&rans0);
    RansEncInit(&rans1);
    RansEncInit(&rans2);
    RansEncInit(&rans3);

    uint8_t* ptr = out_end;

    int isz4 = in_size>>2;
    int i0 = 1*isz4-2;
    int i1 = 2*isz4-2;
    int i2 = 3*isz4-2;
    int i3 = 4*isz4-2;

    unsigned char l0 = in[i0+1];
    unsigned char l1 = in[i1+1];
    unsigned char l2 = in[i2+1];
    unsigned char l3 = in[i3+1];

    // Deal with the remainder
    l3 = in[in_size-1];
    for (i3 = in_size-2; i3 > 4*isz4-2; i3--) {
	unsigned char c3 = in[i3];
	RansEncPutSymbol(&rans3, &ptr, &syms[c3][l3]);
	l3 = c3;
    }

    for (; i0 >= 0; i0--, i1--, i2--, i3--) {
	unsigned char c0, c1, c2, c3;
	RansEncSymbol *s3 = &syms[c3 = in[i3]][l3];
	RansEncSymbol *s2 = &syms[c2 = in[i2]][l2];
	RansEncSymbol *s1 = &syms[c1 = in[i1]][l1];
	RansEncSymbol *s0 = &syms[c0 = in[i0]][l0];

	RansEncPutSymbol(&rans3, &ptr, s3);
	RansEncPutSymbol(&rans2, &ptr, s2);
	RansEncPutSymbol(&rans1, &ptr, s1);
	RansEncPutSymbol(&rans0, &ptr, s0);

	l0 = c0;
	l1 = c1;
	l2 = c2;
	l3 = c3;
    }

    RansEncPutSymbol(&rans3, &ptr, &syms[0][l3]);
    RansEncPutSymbol(&rans2, &ptr, &syms[0][l2]);
    RansEncPutSymbol(&rans1, &ptr, &syms[0][l1]);
    RansEncPutSymbol(&rans0, &ptr, &syms[0][l0]);

    RansEncFlush(&rans3, &ptr);
    RansEncFlush(&rans2, &ptr);
    RansEncFlush(&rans1, &ptr);
    RansEncFlush(&rans0, &ptr);

    *out_size = (out_end - ptr) + tab_size;

    cp = out;
    memmove(out + tab_size, ptr, out_end-ptr);

    return out;
}

#ifndef NO_THREADS
/*
 * Thread local storage per thread in the pool.
 */
pthread_once_t rans_once = PTHREAD_ONCE_INIT;
pthread_key_t rans_key;

static void rans_tls_init(void) {
    pthread_key_create(&rans_key, free);
}
#endif

//#define MAGIC2 111
#define MAGIC2 179
//#define MAGIC2 0
typedef struct {
    uint16_t f;
    uint16_t b;
} fb_t;

static
unsigned char *rans_uncompress_O1_4x16(unsigned char *in, unsigned int in_size,
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
    unsigned char *cp = in, *cp_end = in+in_size, *out_free = NULL;
    unsigned char *c_freq = NULL;
    int i, j = -999;
    unsigned int x;

#ifndef NO_THREADS
    /*
     * The calloc below is expensive as it's a large structure.  We
     * could use malloc, but we're only initialising parts of the structure
     * that we need to, as dictated by the frequency table.  This is far
     * faster than initialising everything (ie malloc+memset => calloc).
     * Not initialising the data means malformed input with mismatching
     * frequency tables to actual data can lead to accessing of the
     * uninitialised sfb table and in turn potential leakage of the
     * uninitialised memory returned by malloc.  That could be anything at
     * all, including important encryption keys used within a server (for
     * example).
     *
     * However (I hope!) we don't care about leaking about the sfb symbol
     * frequencies previously computed by an earlier execution of *this*
     * code.  So calloc once and reuse is the fastest alternative.
     *
     * We do this through pthread local storage as we don't know if this
     * code is being executed in many threads simultaneously.
     */
    pthread_once(&rans_once, rans_tls_init);

    uint8_t *sfb_ = pthread_getspecific(rans_key);
    if (!sfb_) {
	sfb_ = calloc(256*(TOTFREQ_O1+MAGIC2), sizeof(*sfb_));
	pthread_setspecific(rans_key, sfb_);
    }
#else
    uint8_t *sfb_ = calloc(256*(TOTFREQ_O1+MAGIC2), sizeof(*sfb_));
#endif

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
		if (F[j] > (1<<shift) - x)
		    goto err;

		memset(&sfb[i][x], j, F[j]);
		fb[i][j].f = F[j];
		fb[i][j].b = x;
		x += F[j];
	    }
	}
	if (x != (1<<shift))
	    goto err;
    }

    if (tab_end)
	cp = tab_end;
    free(c_freq);
    c_freq = NULL;

    if (cp+16 > cp_end)
	goto err;

    RansState rans0, rans1, rans2, rans3;
    uint8_t *ptr = cp, *ptr_end = in + in_size - 8;
    RansDecInit(&rans0, &ptr); if (rans0 < RANS_BYTE_L) goto err;
    RansDecInit(&rans1, &ptr); if (rans1 < RANS_BYTE_L) goto err;
    RansDecInit(&rans2, &ptr); if (rans2 < RANS_BYTE_L) goto err;
    RansDecInit(&rans3, &ptr); if (rans3 < RANS_BYTE_L) goto err;

    unsigned int isz4 = out_sz>>2;
    int l0 = 0, l1 = 0, l2 = 0, l3 = 0;
    unsigned int i4[] = {0*isz4, 1*isz4, 2*isz4, 3*isz4};

    RansState R[4];
    R[0] = rans0;
    R[1] = rans1;
    R[2] = rans2;
    R[3] = rans3;

    // Around 15% faster to specialise for 10/12 than to have one
    // loop with shift as a variable.
    if (shift == TF_SHIFT_O1) {
	// TF_SHIFT_O1 = 12

	const uint32_t mask = ((1u << TF_SHIFT_O1)-1);
	for (; i4[0] < isz4; i4[0]++, i4[1]++, i4[2]++, i4[3]++) {
	    uint16_t m, c;
	    c = sfb[l0][m = R[0] & mask];
	    R[0] = fb[l0][c].f * (R[0]>>TF_SHIFT_O1) + m - fb[l0][c].b;
	    out[i4[0]] = l0 = c;

	    c = sfb[l1][m = R[1] & mask];
	    R[1] = fb[l1][c].f * (R[1]>>TF_SHIFT_O1) + m - fb[l1][c].b;
	    out[i4[1]] = l1 = c;

	    c = sfb[l2][m = R[2] & mask];
	    R[2] = fb[l2][c].f * (R[2]>>TF_SHIFT_O1) + m - fb[l2][c].b;
	    out[i4[2]] = l2 = c;

	    c = sfb[l3][m = R[3] & mask];
	    R[3] = fb[l3][c].f * (R[3]>>TF_SHIFT_O1) + m - fb[l3][c].b;
	    out[i4[3]] = l3 = c;

	    if (ptr < ptr_end) {
		RansDecRenorm(&R[0], &ptr);
		RansDecRenorm(&R[1], &ptr);
		RansDecRenorm(&R[2], &ptr);
		RansDecRenorm(&R[3], &ptr);
	    } else {
		RansDecRenormSafe(&R[0], &ptr, ptr_end+8);
		RansDecRenormSafe(&R[1], &ptr, ptr_end+8);
		RansDecRenormSafe(&R[2], &ptr, ptr_end+8);
		RansDecRenormSafe(&R[3], &ptr, ptr_end+8);
	    }
	}

	// Remainder
	for (; i4[3] < out_sz; i4[3]++) {
	    uint32_t m3 = R[3] & ((1u<<TF_SHIFT_O1)-1);
	    unsigned char c3 = sfb[l3][m3];
	    out[i4[3]] = c3;
	    R[3] = fb[l3][c3].f * (R[3]>>TF_SHIFT_O1) + m3 - fb[l3][c3].b;
	    RansDecRenormSafe(&R[3], &ptr, ptr_end + 8);
	    l3 = c3;
	}
    } else {
	// TF_SHIFT_O1 = 10
	const uint32_t mask = ((1u << TF_SHIFT_O1_FAST)-1);
	for (; i4[0] < isz4; i4[0]++, i4[1]++, i4[2]++, i4[3]++) {
	    uint16_t m, c;
	    c = sfb[l0][m = R[0] & mask];
	    R[0] = fb[l0][c].f * (R[0]>>TF_SHIFT_O1_FAST) + m - fb[l0][c].b;
	    out[i4[0]] = l0 = c;

	    c = sfb[l1][m = R[1] & mask];
	    R[1] = fb[l1][c].f * (R[1]>>TF_SHIFT_O1_FAST) + m - fb[l1][c].b;
	    out[i4[1]] = l1 = c;

	    c = sfb[l2][m = R[2] & mask];
	    R[2] = fb[l2][c].f * (R[2]>>TF_SHIFT_O1_FAST) + m - fb[l2][c].b;
	    out[i4[2]] = l2 = c;

	    c = sfb[l3][m = R[3] & mask];
	    R[3] = fb[l3][c].f * (R[3]>>TF_SHIFT_O1_FAST) + m - fb[l3][c].b;
	    out[i4[3]] = l3 = c;

	    if (ptr < ptr_end) {
		RansDecRenorm(&R[0], &ptr);
		RansDecRenorm(&R[1], &ptr);
		RansDecRenorm(&R[2], &ptr);
		RansDecRenorm(&R[3], &ptr);
	    } else {
		RansDecRenormSafe(&R[0], &ptr, ptr_end+8);
		RansDecRenormSafe(&R[1], &ptr, ptr_end+8);
		RansDecRenormSafe(&R[2], &ptr, ptr_end+8);
		RansDecRenormSafe(&R[3], &ptr, ptr_end+8);
	    }
	}

	// Remainder
	for (; i4[3] < out_sz; i4[3]++) {
	    uint32_t m3 = R[3] & ((1u<<TF_SHIFT_O1_FAST)-1);
	    unsigned char c3 = sfb[l3][m3];
	    out[i4[3]] = c3;
	    R[3] = fb[l3][c3].f * (R[3]>>TF_SHIFT_O1_FAST) + m3 - fb[l3][c3].b;
	    RansDecRenormSafe(&R[3], &ptr, ptr_end + 8);
	    l3 = c3;
	}
    }
    //fprintf(stderr, "    1 Decoded %d bytes\n", (int)(ptr-in)); //c-size

#ifdef NO_THREADS
    free(sfb_);
#endif
    return out;

 err:
#ifdef NO_THREADS
    free(sfb_);
#endif
    free(out_free);
    free(c_freq);

    return NULL;
}
#endif

/*-----------------------------------------------------------------------------
 * Simple interface to the order-0 vs order-1 encoders and decoders.
 *
 * Smallest is method, <in_size> <input>, so worst case 2 bytes longer.
 */
unsigned char *rans_compress_to_4x16(unsigned char *in,  unsigned int in_size,
				     unsigned char *out, unsigned int *out_size,
				     int order) {
    unsigned int c_meta_len;
    uint8_t *meta = NULL, *rle = NULL, *packed = NULL;

    if (!out) {
	*out_size = rans_compress_bound_4x16(in_size, order);
	if (!(out = malloc(*out_size)))
	    return NULL;

    }
    if (*out_size == 0)
	return NULL;

    unsigned char *out_end = out + *out_size;

    if (in_size <= 20)
	order &= ~X_STRIPE;
    if (in_size <= 1000)
	order &= ~X_32;

    if (order & X_STRIPE) {
	int N = (order>>8);
	if (N == 0) N = 4; // default for compatibility with old tests

	if (N > 255)
	    return NULL;

	unsigned char *transposed = malloc(in_size);
	unsigned int part_len[256];
	unsigned int idx[256];
	if (!transposed)
	    return NULL;
	int i, j, x;

	for (i = 0; i < N; i++) {
	    part_len[i] = in_size / N + ((in_size % N) > i);
	    idx[i] = i ? idx[i-1] + part_len[i-1] : 0; // cumulative index
	}

	for (i = x = 0; i < in_size-N; i += N, x++) {
	    for (j = 0; j < N; j++)
		transposed[idx[j]+x] = in[i+j];
	}
	for (; i < in_size; i += N, x++) {
	    for (j = 0; i+j < in_size; j++)
		transposed[idx[j]+x] = in[i+j];
	}

	unsigned int olen2;
	unsigned char *out2, *out2_start;
	c_meta_len = 1;
	*out = order & ~X_NOSZ;
	c_meta_len += var_put_u32(out+c_meta_len, out_end, in_size);
	out[c_meta_len++] = N;
	
	out2_start = out2 = out+2+5*N; // shares a buffer with c_meta
        for (i = 0; i < N; i++) {
            // Brute force try all methods.
            int j, m[] = {1,64,128,0}, best_j = 0, best_sz = in_size+10;
            for (j = 0; j < 4; j++) {
		if ((order & m[j]) != m[j])
                    continue;
                olen2 = *out_size - (out2 - out);
                rans_compress_to_4x16(transposed+idx[i], part_len[i],
				      out2, &olen2, m[j] | X_NOSZ);
                if (best_sz > olen2) {
                    best_sz = olen2;
                    best_j = j;
                }
            }
	    if (best_j != j-1) {
		olen2 = *out_size - (out2 - out);
		rans_compress_to_4x16(transposed+idx[i], part_len[i],
				      out2, &olen2, m[best_j] | X_NOSZ);
	    }
            out2 += olen2;
            c_meta_len += var_put_u32(out+c_meta_len, out_end, olen2);
        }
	memmove(out+c_meta_len, out2_start, out2-out2_start);
	free(transposed);
	*out_size = c_meta_len + out2-out2_start;
	return out;
    }

    if (order & X_CAT) {
	out[0] = X_CAT;
	c_meta_len = 1;
	c_meta_len += var_put_u32(&out[1], out_end, in_size);
	memcpy(out+c_meta_len, in, in_size);
	*out_size = c_meta_len + in_size;
	return out;
    }

    int do_pack = order & X_PACK;
    int do_rle  = order & X_RLE;
    int no_size = order & X_NOSZ;

    out[0] = order;
    c_meta_len = 1;

    if (!no_size)
	c_meta_len += var_put_u32(&out[1], out_end, in_size);

    order &= 0xf;

    // Format is compressed meta-data, compressed data.
    // Meta-data can be empty, pack, rle lengths, or pack + rle lengths.
    // Data is either the original data, bit-packed packed, rle literals or
    // packed + rle literals.

    if (do_pack && in_size) {
	// PACK 2, 4 or 8 symbols into one byte.
	int pmeta_len;
	uint64_t packed_len;
	packed = hts_pack(in, in_size, out+c_meta_len, &pmeta_len, &packed_len);
	if (!packed || (pmeta_len == 1 && out[c_meta_len] > 16)) {
	    out[0] &= ~X_PACK;
	    do_pack = 0;
	    free(packed);
	    packed = NULL;
	} else {
	    in = packed;
	    in_size = packed_len;
	    c_meta_len += pmeta_len;

	    // Could derive this rather than storing verbatim.
	    // Orig size * 8/nbits (+1 if not multiple of 8/n)
	    int sz = var_put_u32(out+c_meta_len, out_end, in_size);
	    c_meta_len += sz;
	    *out_size -= sz;
	}
    } else if (do_pack) {
	out[0] &= ~X_PACK;
    }

    if (do_rle && in_size) {
	// RLE 'in' -> rle_length + rle_literals arrays
	unsigned int rmeta_len, c_rmeta_len;
	uint64_t rle_len;
	c_rmeta_len = in_size+257;
	if (!(meta = malloc(c_rmeta_len)))
	    return NULL;

	uint8_t rle_syms[256];
	int rle_nsyms = 0;
	uint64_t rmeta_len64;
	rle = rle_encode(in, in_size, meta, &rmeta_len64,
			 rle_syms, &rle_nsyms, NULL, &rle_len);
	memmove(meta+1+rle_nsyms, meta, rmeta_len64);
	meta[0] = rle_nsyms;
	memcpy(meta+1, rle_syms, rle_nsyms);
	rmeta_len = rmeta_len64 + rle_nsyms+1;

	if (!rle || rle_len + rmeta_len >= .99*in_size) {
	    // Not worth the speed hit.
	    out[0] &= ~X_RLE;
	    do_rle = 0;
	    free(rle);
	    rle = NULL;
	} else {
	    // Compress lengths with O0 and literals with O0/O1 ("order" param)
	    int sz = var_put_u32(out+c_meta_len, out_end, rmeta_len*2), sz2;
	    sz += var_put_u32(out+c_meta_len+sz, out_end, rle_len);
	    c_rmeta_len = *out_size - (c_meta_len+sz+5);
	    rans_compress_O0_4x16(meta, rmeta_len, out+c_meta_len+sz+5, &c_rmeta_len);
	    if (c_rmeta_len < rmeta_len) {
		sz2 = var_put_u32(out+c_meta_len+sz, out_end, c_rmeta_len);
		memmove(out+c_meta_len+sz+sz2, out+c_meta_len+sz+5, c_rmeta_len);
	    } else {
		// Uncompressed RLE meta-data as too small
		sz = var_put_u32(out+c_meta_len, out_end, rmeta_len*2+1);
		sz2 = var_put_u32(out+c_meta_len+sz, out_end, rle_len);
		memcpy(out+c_meta_len+sz+sz2, meta, rmeta_len);
		c_rmeta_len = rmeta_len;
	    }

	    c_meta_len += sz + sz2 + c_rmeta_len;

	    in = rle;
	    in_size = rle_len;
	}

	free(meta);
    } else if (do_rle) {
	out[0] &= ~X_RLE;
    }

    *out_size -= c_meta_len;
    if (order && in_size < 8) {
	out[0] &= ~1;
	order  &= ~1;
    }

    if (order == 1)
	rans_compress_O1_4x16(in, in_size, out+c_meta_len, out_size);
    else
	rans_compress_O0_4x16(in, in_size, out+c_meta_len, out_size);

    if (*out_size >= in_size && 0) {
	out[0] &= ~3;
	out[0] |= X_CAT | no_size;
	memcpy(out+c_meta_len, in, in_size);
	*out_size = in_size;
    }

    free(rle);
    free(packed);

    *out_size += c_meta_len;

    return out;
}

unsigned char *rans_compress_4x16(unsigned char *in, unsigned int in_size,
				  unsigned int *out_size, int order) {
    return rans_compress_to_4x16(in, in_size, NULL, out_size, order);
}

unsigned char *rans_uncompress_to_4x16(unsigned char *in,  unsigned int in_size,
				       unsigned char *out, unsigned int *out_size) {
    unsigned char *in_end = in + in_size;
    unsigned char *out_free = NULL, *tmp_free = NULL, *meta_free = NULL;

    if (in_size == 0)
	return NULL;

    if (*in & X_STRIPE) {
	unsigned int ulen, olen, c_meta_len = 1;
	int i;
	uint64_t clen_tot = 0;

	// Decode lengths
	c_meta_len += var_get_u32(in+c_meta_len, in_end, &ulen);
	if (c_meta_len >= in_size)
	    return NULL;
	unsigned int N = in[c_meta_len++];
	unsigned int clenN[256], ulenN[256], idxN[256];
	if (!out) {
	    if (ulen >= INT_MAX)
		return NULL;
	    if (!(out_free = out = malloc(ulen))) {
		return NULL;
	    }
	    *out_size = ulen;
	}
	if (ulen != *out_size) {
	    free(out_free);
	    return NULL;
	}

	for (i = 0; i < N; i++) {
	    ulenN[i] = ulen / N + ((ulen % N) > i);
	    idxN[i] = i ? idxN[i-1] + ulenN[i-1] : 0;
	    c_meta_len += var_get_u32(in+c_meta_len, in_end, &clenN[i]);
	    clen_tot += clenN[i];
	    if (c_meta_len > in_size || clenN[i] > in_size || clenN[i] < 1) {
		free(out_free);
		return NULL;
	    }
	}

	// We can call this with a larger buffer, but once we've determined
	// how much we really use we limit it so the recursion becomes easier
	// to limit.
	if (c_meta_len + clen_tot > in_size) {
	    free(out_free);
	    return NULL;
	}
	in_size = c_meta_len + clen_tot;

	//fprintf(stderr, "    stripe meta %d\n", c_meta_len); //c-size

	// Uncompress the N streams
	unsigned char *outN = malloc(ulen);
	if (!outN) {
	    free(out_free);
	    return NULL;
	}
	for (i = 0; i < N; i++) {
	    olen = ulenN[i];
	    if (in_size < c_meta_len) {
		free(out_free);
		free(outN);
		return NULL;
	    }
	    if (!rans_uncompress_to_4x16(in+c_meta_len, in_size-c_meta_len, outN + idxN[i], &olen)
		|| olen != ulenN[i]) {
		free(out_free);
		free(outN);
		return NULL;
	    }
	    c_meta_len += clenN[i];
	}

	unstripe(out, outN, ulen, N, idxN);

	free(outN);
	*out_size = ulen;
	return out;
    }

    int order = *in++;  in_size--;
    int do_pack = order & X_PACK;
    int do_rle  = order & X_RLE;
    int do_cat  = order & X_CAT;
    int no_size = order & X_NOSZ;
    order &= 1;

    int sz = 0;
    unsigned int osz;
    if (!no_size) {
	sz = var_get_u32(in, in_end, &osz);
    } else
	sz = 0, osz = *out_size;
    in += sz;
    in_size -= sz;

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    if (osz > 100000)
	return NULL;
#endif

    if (no_size && !out)
	goto err; // Need one or the other

    if (!out) {
	*out_size = osz;
	if (!(out = out_free = malloc(*out_size)))
	    return NULL;
    } else {
	if (*out_size < osz)
	goto err;
	*out_size = osz;
    }

//    if (do_pack || do_rle) {
//	in += sz; // size field not needed when pure rANS
//	in_size -= sz;
//    }

    uint32_t c_meta_size = 0;
    unsigned int tmp1_size = *out_size;
    unsigned int tmp2_size = *out_size;
    unsigned int tmp3_size = *out_size;
    unsigned char *tmp1 = NULL, *tmp2 = NULL, *tmp3 = NULL, *tmp = NULL;

    // Need In, Out and Tmp buffers with temporary buffer of the same size
    // as output.  All use rANS, but with optional transforms (none, RLE,
    // Pack, or both).
    //
    //                    rans   unrle  unpack
    // If none:     in -> out
    // If RLE:      in -> tmp -> out
    // If Pack:     in -> tmp        -> out
    // If RLE+Pack: in -> out -> tmp -> out
    //                    tmp1   tmp2   tmp3
    //
    // So rans is in   -> tmp1
    // RLE     is tmp1 -> tmp2
    // Unpack  is tmp2 -> tmp3

    // Format is meta data (Pack and RLE in that order if present),
    // followed by rANS compressed data.

    if (do_pack || do_rle) {
	if (!(tmp = tmp_free = malloc(*out_size)))
	    goto err;
	if (do_pack && do_rle) {
	    tmp1 = out;
	    tmp2 = tmp;
	    tmp3 = out;
	} else if (do_pack) {
	    tmp1 = tmp;
	    tmp2 = tmp1;
	    tmp3 = out;
	} else if (do_rle) {
	    tmp1 = tmp;
	    tmp2 = out;
	    tmp3 = out;
	}
    } else {
	// neither
	tmp  = NULL;
	tmp1 = out;
	tmp2 = out;
	tmp3 = out;
    }

    
    // Decode the bit-packing map.
    uint8_t map[16] = {0};
    int npacked_sym = 0;
    uint64_t unpacked_sz = 0; // FIXME: rename to packed_per_byte
    if (do_pack) {
	c_meta_size = hts_unpack_meta(in, in_size, *out_size, map, &npacked_sym);
	if (c_meta_size == 0)
	    goto err;

	unpacked_sz = osz;
	in      += c_meta_size;
	in_size -= c_meta_size;

	// New unpacked size.  We could derive this bit from *out_size
	// and npacked_sym.
	unsigned int osz;
	sz = var_get_u32(in, in_end, &osz);
	in += sz;
	in_size -= sz;
	if (osz > tmp1_size)
	    goto err;
	tmp1_size = osz;
    }

    uint8_t *meta = NULL;
    uint32_t u_meta_size = 0;
    if (do_rle) {
	// Uncompress meta data
	uint32_t c_meta_size, rle_len, sz;
	sz  = var_get_u32(in,    in_end, &u_meta_size);
	sz += var_get_u32(in+sz, in_end, &rle_len);
	if (rle_len > tmp1_size) // should never grow
	    goto err;
	if (u_meta_size & 1) {
	    meta = in + sz;
	    u_meta_size = u_meta_size/2 > (in_end-meta) ? (in_end-meta) : u_meta_size/2;
	    c_meta_size = u_meta_size;
	} else {
	    sz += var_get_u32(in+sz, in_end, &c_meta_size);
	    u_meta_size /= 2;
	    meta_free = meta = rans_uncompress_O0_4x16(in+sz, in_size-sz, NULL, u_meta_size);
	    if (!meta)
		goto err;
	}
	if (c_meta_size+sz > in_size)
	    goto err;
	in      += c_meta_size+sz;
	in_size -= c_meta_size+sz;
	tmp1_size = rle_len;
    }
   
    //fprintf(stderr, "    meta_size %d bytes\n", (int)(in - orig_in)); //c-size

    // uncompress RLE data.  in -> tmp1
    if (in_size) {
	if (do_cat) {
	    //fprintf(stderr, "    CAT %d\n", tmp1_size); //c-size
	    if (tmp1_size > in_size)
		goto err;
	    if (tmp1_size > *out_size)
		goto err;
	    memcpy(tmp1, in, tmp1_size);
	} else {
	    tmp1 = order
		? rans_uncompress_O1_4x16(in, in_size, tmp1, tmp1_size)
		: rans_uncompress_O0_4x16(in, in_size, tmp1, tmp1_size);
	    if (!tmp1)
		goto err;
	}
    } else {
	tmp1 = NULL;
	tmp1_size = 0;
    }
    tmp2_size = tmp3_size = tmp1_size;

    if (do_rle) {
	// Unpack RLE.  tmp1 -> tmp2.
	if (u_meta_size == 0)
	    goto err;
	uint64_t unrle_size = *out_size;
	int rle_nsyms = *meta ? *meta : 256;
	if (u_meta_size < 1+rle_nsyms)
	    goto err;
	if (!rle_decode(tmp1, tmp1_size,
			meta+1+rle_nsyms, u_meta_size-(1+rle_nsyms),
			meta+1, rle_nsyms, tmp2, &unrle_size))
	    goto err;
	tmp3_size = tmp2_size = unrle_size;
	free(meta_free);
	meta_free = NULL;
    }
    if (do_pack) {
	// Unpack bits via pack-map.  tmp2 -> tmp3
	if (npacked_sym == 1)
	    unpacked_sz = tmp2_size;
	//uint8_t *porig = unpack(tmp2, tmp2_size, unpacked_sz, npacked_sym, map);
	//memcpy(tmp3, porig, unpacked_sz);
	if (!hts_unpack(tmp2, tmp2_size, tmp3, unpacked_sz, npacked_sym, map))
	    goto err;
	tmp3_size = unpacked_sz;
    }

    if (tmp)
	free(tmp);

    *out_size = tmp3_size;
    return tmp3;

 err:
    free(meta_free);
    free(out_free);
    free(tmp_free);
    return NULL;
}

unsigned char *rans_uncompress_4x16(unsigned char *in, unsigned int in_size,
				    unsigned int *out_size) {
    return rans_uncompress_to_4x16(in, in_size, NULL, out_size);
}
