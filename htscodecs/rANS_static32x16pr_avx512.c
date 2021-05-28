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

/*
 * This is an AVX512 implementation of the 32-way interleaved 16-bit rANS.
 * For now it only contains an order-0 implementation.  The AVX2 code may
 * be used for order-1.
 */

#include "config.h"

#ifdef HAVE_AVX512

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

#define NX 32

unsigned char *rans_compress_O0_32x16_avx512(unsigned char *in,
					     unsigned int in_size,
					     unsigned char *out,
					     unsigned int *out_size) {
    unsigned char *cp, *out_end;
    RansEncSymbol syms[256];
    RansState ransN[NX] __attribute__((aligned(64)));
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

    // Build lookup tables for SIMD encoding
    uint32_t SB[256], SA[256], SD[256], SC[256];
    for (i = 0; i < 256; i++) {
	SB[i] = syms[i].x_max;
	SA[i] = syms[i].rcp_freq;
	SD[i] = (syms[i].cmpl_freq<<0) | ((syms[i].rcp_shift-32)<<16);
	SC[i] = syms[i].bias;
    }

#define LOAD512(a,b)					 \
    __m512i a##1 = _mm512_load_si512((__m512i *)&b[0]); \
    __m512i a##2 = _mm512_load_si512((__m512i *)&b[16]);

#define STORE512(a,b)				\
    _mm512_store_si512((__m256i *)&b[0], a##1); \
    _mm512_store_si512((__m256i *)&b[16], a##2);

    LOAD512(Rv, ransN);

    uint16_t *ptr16 = (uint16_t *)ptr;
    for (i=(in_size &~(32-1)); i>0; i-=32) {
	uint8_t *c = &in[i-32];

	// GATHER versions
	// Much faster now we have an efficient loadu mechanism in place,
	// BUT...
	// Try this for avx2 variant too?  Better way to populate the mm256
	// regs for mix of avx2 and avx512 opcodes.
	__m256i c12 = _mm256_loadu_si256((__m256i const *)c);
	__m512i c1 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(c12,0));
	__m512i c2 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(c12,1));
#define SET512(a,b) \
        __m512i a##1 = _mm512_i32gather_epi32(c1, b, 4); \
        __m512i a##2 = _mm512_i32gather_epi32(c2, b, 4)

	SET512(xmax, SB);

	uint16_t gt_mask1 = _mm512_cmpgt_epi32_mask(Rv1, xmax1);
	int pc1 = _mm_popcnt_u32(gt_mask1);
        __m512i Rp1 = _mm512_and_si512(Rv1, _mm512_set1_epi32(0xffff));
        __m512i Rp2 = _mm512_and_si512(Rv2, _mm512_set1_epi32(0xffff));
	uint16_t gt_mask2 = _mm512_cmpgt_epi32_mask(Rv2, xmax2);
	SET512(SDv,  SD);
	int pc2 = _mm_popcnt_u32(gt_mask2);

	//Rp1 = _mm512_maskz_compress_epi32(gt_mask1, Rp1);
	Rp1 = _mm512_maskz_compress_epi32(gt_mask1, Rp1);
	Rp2 = _mm512_maskz_compress_epi32(gt_mask2, Rp2);

	_mm512_mask_cvtepi32_storeu_epi16(ptr16-pc2, (1<<pc2)-1, Rp2);
	ptr16 -= pc2;
	_mm512_mask_cvtepi32_storeu_epi16(ptr16-pc1, (1<<pc1)-1, Rp1);
	ptr16 -= pc1;

	SET512(rfv,  SA);
	Rv1 = _mm512_mask_srli_epi32(Rv1, gt_mask1, Rv1, 16);
	Rv2 = _mm512_mask_srli_epi32(Rv2, gt_mask2, Rv2, 16);

	// interleaved form of this, helps on icc a bit
	//rfv1 = _mm512_mulhi_epu32(Rv1, rfv1);
	//rfv2 = _mm512_mulhi_epu32(Rv2, rfv2);

	// Alternatives here:
	// SHIFT right/left instead of AND: (very marginally slower)
	//   rf1_hm = _mm512_and_epi32(rf1_hm, _mm512_set1_epi64((uint64_t)0xffffffff00000000));
	// vs
	//   rf1_hm = _mm512_srli_epi64(rf1_hm, 32); rf1_hm = _mm512_slli_epi64(rf1_hm, 32);
	__m512i rf1_hm = _mm512_mul_epu32(_mm512_srli_epi64(Rv1, 32), _mm512_srli_epi64(rfv1, 32));
	__m512i rf2_hm = _mm512_mul_epu32(_mm512_srli_epi64(Rv2, 32), _mm512_srli_epi64(rfv2, 32));
	__m512i rf1_lm = _mm512_srli_epi64(_mm512_mul_epu32(Rv1, rfv1), 32);
	__m512i rf2_lm = _mm512_srli_epi64(_mm512_mul_epu32(Rv2, rfv2), 32);
	rf1_hm = _mm512_and_epi32(rf1_hm, _mm512_set1_epi64((uint64_t)0xffffffff00000000));
	rf2_hm = _mm512_and_epi32(rf2_hm, _mm512_set1_epi64((uint64_t)0xffffffff00000000));
	rfv1 = _mm512_or_epi32(rf1_lm, rf1_hm);
	rfv2 = _mm512_or_epi32(rf2_lm, rf2_hm);

	// Or a pure masked blend approach, sadly slower.
	// rfv1 = _mm512_mask_blend_epi32(0x5555, _mm512_mul_epu32(_mm512_srli_epi64(Rv1, 32), _mm512_srli_epi64(rfv1, 32)), _mm512_srli_epi64(_mm512_mul_epu32(Rv1, rfv1), 32));
	// rfv2 = _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64(_mm512_mul_epu32(Rv2, rfv2), 32), _mm512_mul_epu32(_mm512_srli_epi64(Rv2, 32), _mm512_srli_epi64(rfv2, 32)));

	SET512(biasv, SC);
	__m512i shiftv1 = _mm512_srli_epi32(SDv1, 16);
	__m512i shiftv2 = _mm512_srli_epi32(SDv2, 16);

	__m512i qv1 = _mm512_srlv_epi32(rfv1, shiftv1);
	__m512i qv2 = _mm512_srlv_epi32(rfv2, shiftv2);

	qv1 = _mm512_mullo_epi32(qv1, _mm512_and_si512(SDv1, _mm512_set1_epi32(0xffff)));
	qv1 = _mm512_add_epi32(qv1, biasv1);
	Rv1 = _mm512_add_epi32(Rv1, qv1);

	qv2 = _mm512_mullo_epi32(qv2, _mm512_and_si512(SDv2, _mm512_set1_epi32(0xffff)));
	qv2 = _mm512_add_epi32(qv2, biasv2);
	Rv2 = _mm512_add_epi32(Rv2, qv2);
    }
    ptr = (uint8_t *)ptr16;
    STORE512(Rv, ransN);

    for (z = NX-1; z >= 0; z--)
	RansEncFlush(&ransN[z], &ptr);
    
 empty:
    // Finalise block size and return it
    *out_size = (out_end - ptr) + tab_size;

    memmove(out + tab_size, ptr, out_end-ptr);

    return out;
}

unsigned char *rans_uncompress_O0_32x16_avx512(unsigned char *in,
					       unsigned int in_size,
					       unsigned char *out,
					       unsigned int out_sz) {
    if (in_size < 32*4) // 32-states at least
	return NULL;

    if (out_sz >= INT_MAX)
	return NULL; // protect against some overflow cases

    /* Load in the static tables */
    unsigned char *cp = in, *out_free = NULL;
    unsigned char *cp_end = in + in_size - 8; // within 8 => be extra safe
    int i, j;
    unsigned int x, y;
    uint8_t  ssym [TOTFREQ+64]; // faster to use 16-bit on clang
    uint32_t s3[TOTFREQ]  __attribute__((aligned(64))); // For TF_SHIFT <= 12

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

    int z;
    RansState Rv[NX] __attribute__((aligned(64)));
    for (z = 0; z < NX; z++) {
	RansDecInit(&Rv[z], &cp);
	if (Rv[z] < RANS_BYTE_L)
	    goto err;
    }

    uint16_t *sp = (uint16_t *)cp;

    int out_end = (out_sz&~(NX-1));
    const uint32_t mask = (1u << TF_SHIFT)-1;

    __m512i maskv = _mm512_set1_epi32(mask); // set mask in all lanes
    __m512i R1 = _mm512_load_epi32(&Rv[0]);
    __m512i R2 = _mm512_load_epi32(&Rv[16]);

    // Start of the first loop iteration, which we do move to the end of the
    // loop for the next cycle so we can remove some of the instr. latency.
    __m512i masked1 = _mm512_and_epi32(R1, maskv);
    __m512i masked2 = _mm512_and_epi32(R2, maskv);
    __m512i S1 = _mm512_i32gather_epi32(masked1, (int *)s3, sizeof(*s3));
    __m512i S2 = _mm512_i32gather_epi32(masked2, (int *)s3, sizeof(*s3));

    int offset=0;
    for (i=0; i < out_end; i+=NX) {
      //for (z = 0; z < 16; z++) {
      //uint32_t S = s3[R[z] & mask];
      __m512i renorm_words1 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) (sp+offset))); // next 16 words

      //uint16_t f = S>>(TF_SHIFT+8), b = (S>>8) & mask;
      __m512i f1 = _mm512_srli_epi32(S1, TF_SHIFT+8);
      __m512i f2 = _mm512_srli_epi32(S2, TF_SHIFT+8);
      __m512i b1 = _mm512_and_epi32(_mm512_srli_epi32(S1, 8), maskv);
      __m512i b2 = _mm512_and_epi32(_mm512_srli_epi32(S2, 8), maskv);

      //R[z] = f * (R[z] >> TF_SHIFT) + b;
      // approx 10 cycle latency on mullo.
      R1 = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32(R1, TF_SHIFT), f1), b1);
      R2 = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32(R2, TF_SHIFT), f2), b2);

      // renorm. this is the interesting part:
      __mmask16 renorm_mask1 = _mm512_cmplt_epu32_mask(R1, _mm512_set1_epi32(RANS_BYTE_L));
      __mmask16 renorm_mask2 = _mm512_cmplt_epu32_mask(R2, _mm512_set1_epi32(RANS_BYTE_L));
      offset += _mm_popcnt_u32(renorm_mask1); // advance by however many words we actually read
      __m512i renorm_words2 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) (sp+offset)));

      __m512i renorm_vals1 = _mm512_maskz_expand_epi32(renorm_mask1, renorm_words1); // select masked only
      __m512i renorm_vals2 = _mm512_maskz_expand_epi32(renorm_mask2, renorm_words2); // select masked only
      R1 = _mm512_mask_slli_epi32(R1, renorm_mask1, R1, 16); // shift & add selected words
      R2 = _mm512_mask_slli_epi32(R2, renorm_mask2, R2, 16); // shift & add selected words
      R1 = _mm512_add_epi32(R1, renorm_vals1);
      R2 = _mm512_add_epi32(R2, renorm_vals2);

      // For start of next loop iteration.  This has been moved here
      // (and duplicated to before the loop starts) so we can do something
      // with the latency period of gather, such as finishing up the
      // renorm offset and writing the results. 
      __m512i S1_ = S1; // temporary copy for use in out[]=S later
      __m512i S2_ = S2;

      masked1 = _mm512_and_epi32(R1, maskv);
      masked2 = _mm512_and_epi32(R2, maskv);
      // Gather is slow bit (half total time) - 30 cycle latency.
      S1 = _mm512_i32gather_epi32(masked1, (int *)s3, sizeof(*s3));
      S2 = _mm512_i32gather_epi32(masked2, (int *)s3, sizeof(*s3));

      offset += _mm_popcnt_u32(renorm_mask2); // advance by however many words we actually read

      //out[i+z] = S;
      _mm_storeu_si128((__m128i *)(out+i),    _mm512_cvtepi32_epi8(S1_));
      _mm_storeu_si128((__m128i *)(out+i+16), _mm512_cvtepi32_epi8(S2_));
    }      

    _mm512_store_epi32(&Rv[ 0], R1);
    _mm512_store_epi32(&Rv[16], R2);

    for (z = out_sz & (NX-1); z-- > 0; )
      out[out_end + z] = ssym[Rv[z] & mask];

    return out;

 err:
    free(out_free);
    return NULL;
}

#endif // HAVE_AVX512
