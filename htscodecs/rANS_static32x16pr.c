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

#include "rANS_word.h"
#include "rANS_static4x16.h"
#include "rANS_static16_int.h"
#include "varint.h"
#include "pack.h"
#include "rle.h"
#include "utils.h"

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


#ifndef NO_THRADS
// Reuse the rANS_static4x16pr variables
extern pthread_once_t rans_once;
extern pthread_key_t rans_key;

extern void rans_tls_init(void);
#endif

#define NX 32

// Hist8 with a crude entropy (bits / byte) estimator.
static inline
double hist8e(unsigned char *in, unsigned int in_size, uint32_t F0[256]) {
    uint32_t F1[256+MAGIC] = {0}, F2[256+MAGIC] = {0}, F3[256+MAGIC] = {0};
    uint32_t F4[256+MAGIC] = {0}, F5[256+MAGIC] = {0}, F6[256+MAGIC] = {0};
    uint32_t F7[256+MAGIC] = {0};

#ifdef __GNUC__
    double e = 0, in_size_r2 = log(1.0/in_size)/log(2);
#else
    double e = 0, in_size_r2 = log(1.0/in_size);
#endif

    unsigned int i, i8 = in_size & ~7;
    for (i = 0; i < i8; i+=8) {
	F0[in[i+0]]++;
	F1[in[i+1]]++;
	F2[in[i+2]]++;
	F3[in[i+3]]++;
	F4[in[i+4]]++;
	F5[in[i+5]]++;
	F6[in[i+6]]++;
	F7[in[i+7]]++;
    }
    while (i < in_size)
	F0[in[i++]]++;

    for (i = 0; i < 256; i++) {
	F0[i] += F1[i] + F2[i] + F3[i] + F4[i] + F5[i] + F6[i] + F7[i];
#ifdef __GNUC__
	e -= F0[i] * (32 - __builtin_clz(F0[i]) + in_size_r2);
#else
	extern double fast_log(double);
	e -= F0[i] * (fast_log(F0[i]) + in_size_r2);
#endif
    }

#ifndef __GNUC__
    e /= log(2);
#endif
    return e/in_size;
}

#ifdef __ARM_NEON
#  include "rANS_static32x16pr_neon.c"
#endif

unsigned char *rans_compress_O0_32x16(unsigned char *in, unsigned int in_size,
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
    double e = hist8e(in, in_size, F);
    int low_ent = e < 2;
    //hist8(in, in_size, F); int low_ent = 0;

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

    if (low_ent) {
	for (i=(in_size &~(NX-1)); i>0; i-=NX) {
	    for (z = NX-1; z >= 0; z-=4) {
		RansEncSymbol *s0 = &syms[in[i-(NX-z+0)]];
		RansEncSymbol *s1 = &syms[in[i-(NX-z+1)]];
		RansEncSymbol *s2 = &syms[in[i-(NX-z+2)]];
		RansEncSymbol *s3 = &syms[in[i-(NX-z+3)]];
		RansEncPutSymbol_branched(&ransN[z-0], &ptr, s0);
		RansEncPutSymbol_branched(&ransN[z-1], &ptr, s1);
		RansEncPutSymbol_branched(&ransN[z-2], &ptr, s2);
		RansEncPutSymbol_branched(&ransN[z-3], &ptr, s3);
		if (NX%8 == 0) {
		    z -= 4;
		    RansEncSymbol *s0 = &syms[in[i-(NX-z+0)]];
		    RansEncSymbol *s1 = &syms[in[i-(NX-z+1)]];
		    RansEncSymbol *s2 = &syms[in[i-(NX-z+2)]];
		    RansEncSymbol *s3 = &syms[in[i-(NX-z+3)]];
		    RansEncPutSymbol_branched(&ransN[z-0], &ptr, s0);
		    RansEncPutSymbol_branched(&ransN[z-1], &ptr, s1);
		    RansEncPutSymbol_branched(&ransN[z-2], &ptr, s2);
		    RansEncPutSymbol_branched(&ransN[z-3], &ptr, s3);
		}
	    }
	    if (z < -1) abort();
	}
    } else {
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

unsigned char *rans_uncompress_O0_32x16(unsigned char *in, unsigned int in_size,
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
		ssym [y + x] = j; // needed?
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


//-----------------------------------------------------------------------------

unsigned char *rans_compress_O1_32x16(unsigned char *in, unsigned int in_size,
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
    if (0) {
	// skew stats to never get max freq of 4096.
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

//#define MAGIC2 111
#define MAGIC2 179
//#define MAGIC2 0
typedef struct {
    uint16_t f;
    uint16_t b;
} fb_t;

unsigned char *rans_uncompress_O1_32x16(unsigned char *in, unsigned int in_size,
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
    uint32_t s3[256][TOTFREQ_O1_FAST];

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

		if (shift == TF_SHIFT_O1_FAST) {
		    int y;
		    for (y = 0; y < F[j]; y++)
			s3[i][y+x] = (((uint32_t)F[j])<<(shift+8)) |(y<<8) |j;
		} else {
		    memset(&sfb[i][x], j, F[j]);
		    fb[i][j].f = F[j];
		    fb[i][j].b = x;
		}

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
		// Merged sfb and fb into single s3 lookup.
		// The m[4] array completely vanishes in this method.
		uint32_t S[4] = {
		    s3[l[z+0]][R[z+0] & mask],
		    s3[l[z+1]][R[z+1] & mask],
		    s3[l[z+2]][R[z+2] & mask],
		    s3[l[z+3]][R[z+3] & mask],
		};

		l[z+0] = out[i4[z+0]++] = S[0];
		l[z+1] = out[i4[z+1]++] = S[1];
		l[z+2] = out[i4[z+2]++] = S[2];
		l[z+3] = out[i4[z+3]++] = S[3];

		uint32_t F[4] = {
		    S[0]>>(TF_SHIFT_O1_FAST+8),
		    S[1]>>(TF_SHIFT_O1_FAST+8),
		    S[2]>>(TF_SHIFT_O1_FAST+8),
		    S[3]>>(TF_SHIFT_O1_FAST+8),
	        };
		uint32_t B[4] = {
		    (S[0]>>8) & mask,
		    (S[1]>>8) & mask,
		    (S[2]>>8) & mask,
		    (S[3]>>8) & mask,
	        };

		R[z+0] = F[0] * (R[z+0]>>TF_SHIFT_O1_FAST) + B[0];
		R[z+1] = F[1] * (R[z+1]>>TF_SHIFT_O1_FAST) + B[1];
		R[z+2] = F[2] * (R[z+2]>>TF_SHIFT_O1_FAST) + B[2];
		R[z+3] = F[3] * (R[z+3]>>TF_SHIFT_O1_FAST) + B[3];

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
	    uint32_t S = s3[l[NX-1]][R[NX-1] & ((1u<<TF_SHIFT_O1_FAST)-1)];
	    out[i4[NX-1]] = l[NX-1] = S&0xff;
	    R[NX-1] = (S>>(TF_SHIFT_O1_FAST+8)) * (R[NX-1]>>TF_SHIFT_O1_FAST)
		+ ((S>>8) & ((1u<<TF_SHIFT_O1_FAST)-1));
	    RansDecRenormSafe(&R[NX-1], &ptr, ptr_end + 8);
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
