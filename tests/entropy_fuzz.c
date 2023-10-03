/* Fuzz testing target. */
/*
 * Copyright (c) 2023 Genome Research Ltd.
 * Author(s): James Bonfield, Rob Davies
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
#include <fcntl.h>
#include <sys/time.h>
#ifndef _WIN32
#include <sys/resource.h>
#include <pthread.h>
#endif

#include "htscodecs/arith_dynamic.h"
#include "htscodecs/rANS_static.h"
#include "htscodecs/rANS_static4x16.h"

int LLVMFuzzerTestOneInput(uint8_t *in, size_t in_size) {
    uint32_t csize, usize;
    int result = 0;

    if (!in_size)
        return 0;

    int order_a[] = {0,1,                            // r4x8
                     64,65, 128,129, 192,193,        // r4x16, arith
                     4,5, 68,69, 132,133, 194,197,   // r4x16 SIMD
                     };
    int i, j;
    for (i = 0; i < sizeof(order_a) / sizeof(*order_a); i++) {
        int order = order_a[i];
        uint8_t *comp, *uncomp;
        for (j = 0; j < 4; j++) {
            int chigh = 4, clow = 0, c;
            uint8_t *comp0 = NULL;
            uint32_t csize0 = 0;
            for (c = 0; c < 4; c+=(j==2)?1:4) {
                // Test combinations of SIMD implementations
                uint32_t chex = (clow<<8) | chigh;
                clow  = 1<<c;
                chigh >>= 1;
                rans_set_cpu(chex);

                // encode
                switch (j) {
                case 0: // r4x8
                    if (i >= 2) continue;
                    comp = rans_compress(in, in_size, &csize, order);
                    break;

                case 1: // r4x16
                    if (i >= 8) continue;
                    comp = rans_compress_4x16(in, in_size, &csize, order);
                    break;

                case 2: // r32x16
                    if (i < 8) continue;
                    comp = rans_compress_4x16(in, in_size, &csize, order);
                    break;

                case 3: // arith
                    if (i >= 8) continue;
                    comp = arith_compress(in, in_size, &csize, order);
                    break;
                }

                if (comp0) {
                    if (csize != csize0 || memcmp(comp, comp0, csize) != 0) {
                        printf("\tFAIL (comp)\n");
                        abort();
                    }
                } else {
                    csize0 = csize;
                    comp0 = comp;
                }

                // decode
                switch (j) {
                case 0: // r4x8
                    if (i >= 2) continue;
                    uncomp = rans_uncompress(comp, csize, &usize);
                    break;

                case 1: // r4x16
                    if (i >= 8) continue;
                    uncomp = rans_uncompress_4x16(comp, csize, &usize);
                    break;

                case 2: // r32x16
                    if (i < 8) continue;
                    uncomp = rans_uncompress_4x16(comp, csize, &usize);
                    break;

                case 3: // arith
                    if (i >= 8) continue;
                    uncomp = arith_uncompress(comp, csize, &usize);
                    break;
                }

                if (usize != in_size || memcmp(in, uncomp, usize) != 0) {
                    printf("\tFAIL\n");
                    abort();
                }

                if (comp != comp0)
                    free(comp);
                free(uncomp);
            }
            free(comp0);
        }
    }

    return result;
}
