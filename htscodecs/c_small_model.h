/*
 * Copyright (c) 2012, 2018-2019, 2022 Genome Research Ltd.
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
 * A specialised version of c_simple_model for small numbers of symbols.
 * This doesn't have the symbol sorting and simply has a direct lookup.
 */

#include <stdint.h>
#include "c_range_coder.h"

/*
 *--------------------------------------------------------------------------
 * A simple frequency model.
 *
 * Define NSYM to be an integer value before including this file.
 * It will then generate types and functions specific to that
 * maximum number of symbols.
 *
 * This keeps a list of symbols and their frequencies, approximately
 * sorted by symbol frequency. We allow for a single symbol to periodically
 * move up the list when emitted, effectively doing a single step of
 * bubble sort periodically. This means it's largely the same complexity
 * irrespective of alphabet size.
 * It's more efficient on strongly biased distributions than random data.
 *
 * There is no escape symbol, so the model is tailored to relatively
 * stationary samples (although we do have occasional normalisation to
 * avoid frequency counters getting too high).
 *--------------------------------------------------------------------------
 */

//-----------------------------------------------------------------------------
// Bits we want included once only - constants, types, etc
#ifndef C_SMALL_MODEL_H
#define C_SMALL_MODEL_H

// MAX_FREQ of 255 permits us to use uint8_t for the frequencies, further
// reducing the model size.  Combined with low NSYM this makes prefetch
// particularly good at precching upcoming multiple models.
#ifdef MAX_FREQ
#  undef MAX_FREQ
#endif
#define PASTE3(a,b,c) a##b##c
#define SMALL_MODEL(a,b) PASTE3(SMALL,a,b)
#ifndef STEP
#    define STEP 1
#endif
#define MAX_FREQ (256-STEP)
#endif /* C_SMALL_MODEL_H */


//-----------------------------------------------------------------------------
// Bits we regenerate for each NSYM value.

typedef struct {
    uint8_t F[NSYM];
} SMALL_MODEL(NSYM,_);


static inline void SMALL_MODEL(NSYM,_init)(SMALL_MODEL(NSYM,_) *m) {
    int i;
    
    for (i=0; i<NSYM; i++)
        m->F[i] = 1;
}


static inline void SMALL_MODEL(NSYM,_normalize)(SMALL_MODEL(NSYM,_) *m) {
    for (int i = 0; i < NSYM; i++)
        m->F[i] -= m->F[i]>>1;
}

// Encode a symbol
static inline void SMALL_MODEL(NSYM,_encodeSymbol)
    (SMALL_MODEL(NSYM,_) *m, RangeCoder *rc, uint16_t sym) {
    int i, tot = 0, acc[NSYM];
    for (i = 0; i < NSYM; i++) {
        acc[i] = tot;
        tot += m->F[i];
    }

    RC_Encode(rc, acc[sym], m->F[sym], tot);
    m->F[sym] += STEP;

    if (tot >= MAX_FREQ)
        SMALL_MODEL(NSYM,_normalize)(m);
}

// Update model frequencies without encoding anything.
static inline void SMALL_MODEL(NSYM,_updateSymbol)
    (SMALL_MODEL(NSYM,_) *m, uint16_t sym) {
    int i, tot = 0;
    for (i = 0; i < NSYM; i++)
        tot += m->F[i];

    m->F[sym] += STEP;

    if (tot >= MAX_FREQ)
        SMALL_MODEL(NSYM,_normalize)(m);
}

// Decode a symbol
static inline uint16_t SMALL_MODEL(NSYM,_decodeSymbol)
    (SMALL_MODEL(NSYM,_) *m, RangeCoder *rc) {
    int sym, tot = 0;
    for (sym = 0; sym < NSYM; sym++)
        tot += m->F[sym];

    uint32_t freq = RC_GetFreq(rc, tot);
    uint32_t acc;
    for (sym = acc = 0; (acc += m->F[sym]) <= freq; sym++)
        ;
    acc -= m->F[sym];

    RC_Decode(rc, acc, m->F[sym], tot);
    m->F[sym] += STEP;

    if (tot >= MAX_FREQ)
        SMALL_MODEL(NSYM,_normalize)(m);

    return sym;
}
