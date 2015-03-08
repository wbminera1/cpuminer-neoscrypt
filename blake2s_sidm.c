/*
 * Copyright 2015 gerko.deroo@kangaderoo.nl
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/* BLAKE2s */

#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#if defined(__x86_64__)

#define BLAKE2S_BLOCK_SIZE_SIDM    64U

typedef struct blake2s_state_sidm_t {
    uint32_t  h[8];
    uint32_t  t[2];
    uint32_t  f[2];
    u_char buf[2 * BLAKE2S_BLOCK_SIZE_SIDM];
    uint32_t  buflen;
} blake2s_state_sidm;

static const uint32_t blake2s_IV_sidm[8] __attribute__((aligned(32))) = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};


static inline void blake2_round_sidm(__m128i* state, __m128i* Message)
{
	__m128i _calc;
	__m128i const _rotate_16   = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
//	__m128i const _rotate_8_l  = _mm_setr_epi8(3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14);
	__m128i const _rotate_8_r  = _mm_setr_epi8(1,2,3,0,5,6,7,4,9,10,11,8,13,14,15,12);

	/* first row */
	state[0] = _mm_add_epi32(state[0], state[1]);
	state[0] = _mm_add_epi32(state[0], Message[0]);
	state[3] = _mm_xor_si128(state[3], state[0]);
	state[3] =  _mm_shuffle_epi8(state[3],_rotate_16);

		/* second row */
	state[2] = _mm_add_epi32(state[2], state[3]);
	_calc = _mm_xor_si128(state[1], state[2]);
	state[1] = _mm_srli_epi32(_calc, (12));
	_calc = _mm_slli_epi32(_calc,(32 - 12));
	state[1] = _mm_xor_si128(state[1], _calc);

		/* third row */
	state[0] = _mm_add_epi32(state[0], state[1]);
	state[0] = _mm_add_epi32(state[0], Message[1]);
	state[3] = _mm_xor_si128(state[3], state[0]);
	state[3] =  _mm_shuffle_epi8(state[3],_rotate_8_r);

		/* fourth row */
	state[2] = _mm_add_epi32(state[2], state[3]);
	_calc = _mm_xor_si128(state[1], state[2]);
	state[1] = _mm_srli_epi32(_calc, (7));
	_calc = _mm_slli_epi32(_calc,(32 - 7));
	state[1] = _mm_xor_si128(state[1], _calc);

// row a		0  1  2  3        0  1  2  3
// row b		4  5  6  7   -->  5  6  7  4
// row c		8  9 10 11   ..> 10 11  8  9
// row d 	   12 13 14 15       15 12 13 14
	// transpose_matrix(row1, row2, row3, row4, row_to_column);
	state[1] = _mm_shuffle_epi32(state[1],0x39); // 10 01 00 11
	state[2] = _mm_shuffle_epi32(state[2],0x4e); // 01 00 11 10
	state[3] = _mm_shuffle_epi32(state[3],0x93); // 00 11 10 01
	// end transpose

		/* first column */
	state[0] = _mm_add_epi32(state[0], state[1]);
	state[0] = _mm_add_epi32(state[0], Message[2]);
	state[3] = _mm_xor_si128(state[3], state[0]);
	state[3] =  _mm_shuffle_epi8(state[3],_rotate_16);

		/* second column */
	state[2] = _mm_add_epi32(state[2], state[3]);
	_calc = _mm_xor_si128(state[1], state[2]);
	state[1] = _mm_srli_epi32(_calc, (12));
	_calc = _mm_slli_epi32(_calc,(32 - 12));
	state[1] = _mm_xor_si128(state[1], _calc);

		/* third column */
	state[0] = _mm_add_epi32(state[0], state[1]);
	state[0] = _mm_add_epi32(state[0], Message[3]);
	state[3] = _mm_xor_si128(state[3], state[0]);
	state[3] =  _mm_shuffle_epi8(state[3],_rotate_8_r);

		/* fourth column */
	state[2] = _mm_add_epi32(state[2], state[3]);
	_calc = _mm_xor_si128(state[1], state[2]);
	state[1] = _mm_srli_epi32(_calc, (7));
	_calc = _mm_slli_epi32(_calc,(32 - 7));
	state[1] = _mm_xor_si128(state[1], _calc);

		// transpose_matrix(row1, row2, row3, row4, row_to_column);
	state[1] = _mm_shuffle_epi32(state[1],0x93);
	state[2] = _mm_shuffle_epi32(state[2],0x4e);
	state[3] = _mm_shuffle_epi32(state[3],0x39);
	// end transpose

}

void blake2_compress_sidm(blake2s_state_sidm *S)
{
	uint32_t v[16] __attribute__((aligned(32)));
	uint32_t *m = (uint *) S->buf;
	__m128i *_mm_v = (__m128i*) &v;
	__m128i Message[4];

	    v[0]  = S->h[0];
	    v[1]  = S->h[1];
	    v[2]  = S->h[2];
	    v[3]  = S->h[3];
	    v[4]  = S->h[4];
	    v[5]  = S->h[5];
	    v[6]  = S->h[6];
	    v[7]  = S->h[7];
	    v[8]  = blake2s_IV_sidm[0];
	    v[9]  = blake2s_IV_sidm[1];
	    v[10] = blake2s_IV_sidm[2];
	    v[11] = blake2s_IV_sidm[3];
	    v[12] = S->t[0] ^ blake2s_IV_sidm[4];
	    v[13] = S->t[1] ^ blake2s_IV_sidm[5];
	    v[14] = S->f[0] ^ blake2s_IV_sidm[6];
	    v[15] = S->f[1] ^ blake2s_IV_sidm[7];

	    // round 1
	    // 	    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,

	    Message[0]=_mm_setr_epi32(m[0], m[2],  m[4], m[6]);
	    Message[1]=_mm_setr_epi32(m[1], m[3],  m[5], m[7]);
	    Message[2]=_mm_setr_epi32(m[8], m[10], m[12], m[14]);
	    Message[3]=_mm_setr_epi32(m[9], m[11], m[13], m[15]);

	    blake2_round_sidm((__m128i *)v, Message);

//	    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
	    Message[0]=_mm_setr_epi32(m[14], m[4],  m[9], m[13]);
	    Message[1]=_mm_setr_epi32(m[10], m[8],  m[15], m[6]);
	    Message[2]=_mm_setr_epi32(m[1], m[0], m[11], m[5]);
	    Message[3]=_mm_setr_epi32(m[12], m[2], m[7], m[3]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
	    Message[0]=_mm_setr_epi32(m[11], m[12],  m[5], m[15]);
	    Message[1]=_mm_setr_epi32(m[8], m[0],  m[2], m[13]);
	    Message[2]=_mm_setr_epi32(m[10], m[3], m[7], m[9]);
	    Message[3]=_mm_setr_epi32(m[14], m[6], m[1], m[4]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
	    Message[0]=_mm_setr_epi32(m[7], m[3],  m[13], m[11]);
	    Message[1]=_mm_setr_epi32(m[9], m[1],  m[12], m[14]);
	    Message[2]=_mm_setr_epi32(m[2], m[5], m[4], m[15]);
	    Message[3]=_mm_setr_epi32(m[6], m[10], m[0], m[8]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
	    Message[0]=_mm_setr_epi32(m[9], m[5],  m[2], m[10]);
	    Message[1]=_mm_setr_epi32(m[0], m[7],  m[4], m[15]);
	    Message[2]=_mm_setr_epi32(m[14], m[11], m[6], m[3]);
	    Message[3]=_mm_setr_epi32(m[1], m[12], m[8], m[13]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
	    Message[0]=_mm_setr_epi32(m[2], m[6],  m[0], m[8]);
	    Message[1]=_mm_setr_epi32(m[12], m[10],  m[11], m[3]);
	    Message[2]=_mm_setr_epi32(m[4], m[7], m[15], m[1]);
	    Message[3]=_mm_setr_epi32(m[13], m[5], m[14], m[9]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
	    Message[0]=_mm_setr_epi32(m[12], m[1],  m[14], m[4]);
	    Message[1]=_mm_setr_epi32(m[5], m[15],  m[13], m[10]);
	    Message[2]=_mm_setr_epi32(m[0], m[6], m[9], m[8]);
	    Message[3]=_mm_setr_epi32(m[7], m[3], m[2], m[11]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
	    Message[0]=_mm_setr_epi32(m[13], m[7],  m[12], m[3]);
	    Message[1]=_mm_setr_epi32(m[11], m[14],  m[1], m[9]);
	    Message[2]=_mm_setr_epi32(m[5], m[15], m[8], m[2]);
	    Message[3]=_mm_setr_epi32(m[0], m[4], m[6], m[10]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
	    Message[0]=_mm_setr_epi32(m[6], m[14],  m[11], m[0]);
	    Message[1]=_mm_setr_epi32(m[15], m[9],  m[3], m[8]);
	    Message[2]=_mm_setr_epi32(m[12], m[13], m[1], m[10]);
	    Message[3]=_mm_setr_epi32(m[2], m[7], m[4], m[5]);

	    blake2_round_sidm((__m128i *)v, Message);
//	    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
	    Message[0]=_mm_setr_epi32(m[10], m[8],  m[7], m[1]);
	    Message[1]=_mm_setr_epi32(m[2], m[4],  m[6], m[5]);
	    Message[2]=_mm_setr_epi32(m[15], m[9], m[3], m[13]);
	    Message[3]=_mm_setr_epi32(m[11], m[14], m[12], m[0]);

	    blake2_round_sidm((__m128i *)v, Message);

	    _mm_v[0] = _mm_xor_si128(_mm_v[0],_mm_v[2]);
	    _mm_v[1] = _mm_xor_si128(_mm_v[1],_mm_v[3]);

	    S->h[0] ^= v[0];// ^ v[8];
	    S->h[1] ^= v[1];// ^ v[9];
	    S->h[2] ^= v[2];// ^ v[10];
	    S->h[3] ^= v[3];// ^ v[11];
	    S->h[4] ^= v[4];// ^ v[12];
	    S->h[5] ^= v[5];// ^ v[13];
	    S->h[6] ^= v[6];// ^ v[14];
	    S->h[7] ^= v[7];// ^ v[15];
}

#endif



