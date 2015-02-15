/*
 * Copyright 2013 gerko.deroo@kangaderoo.nl
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

#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#if (AVX)

static inline void xor_chacha_sidm(__m128i *calc_16, __m128i *calc_12, __m128i *calc_8, __m128i *calc_7,
 								  __m128i *calc_1, __m128i *calc_2, __m128i *calc_3, __m128i *calc_4,
 								  uint32_t double_rounds)
{
	int i;
	__m128i _calc;
	__m128i _rotate_16 = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
	__m128i _rotate_8  = _mm_setr_epi8(3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14);
	__m128i rowa = _mm_xor_si128(*calc_16, *calc_1);;
	__m128i rowb = _mm_xor_si128(*calc_12, *calc_2);;
	__m128i rowc = _mm_xor_si128(*calc_8, *calc_3);;
	__m128i rowd = _mm_xor_si128(*calc_7, *calc_4);;

	*calc_16 = _mm_xor_si128(*calc_16, *calc_1);
	*calc_12 = _mm_xor_si128(*calc_12, *calc_2);
	*calc_8 = _mm_xor_si128(*calc_8, *calc_3);
	*calc_7 = _mm_xor_si128(*calc_7, *calc_4);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd =  _mm_shuffle_epi8(rowd,_rotate_16);

		/* second row */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (12));
		_calc = _mm_srli_epi32(_calc,(32 - 12));
		rowb = _mm_xor_si128(rowb, _calc);

		/* third row */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd =  _mm_shuffle_epi8(rowd,_rotate_8);

		/* fourth row */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (7));
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		rowb = _mm_xor_si128(rowb, _calc);

// row a		0  1  2  3        0  1  2  3
// row b		4  5  6  7   -->  5  6  7  4
// row c		8  9 10 11   ..> 10 11  8  9
// row d 	   12 13 14 15       15 12 13 14
	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb = _mm_shuffle_epi32(rowb,0x39); // 10 01 00 11
		rowc = _mm_shuffle_epi32(rowc,0x4e); // 01 00 11 10
		rowd = _mm_shuffle_epi32(rowd,0x93); // 00 11 10 01
	// end transpose

		/* first column */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd =  _mm_shuffle_epi8(rowd,_rotate_16);

		/* second column */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (12));
		_calc = _mm_srli_epi32(_calc,(32 - 12));
		rowb = _mm_xor_si128(rowb, _calc);

		/* third column */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd =  _mm_shuffle_epi8(rowd,_rotate_8);

		/* fourth column */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (7));
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		rowb = _mm_xor_si128(rowb, _calc);

		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb = _mm_shuffle_epi32(rowb,0x93);
		rowc = _mm_shuffle_epi32(rowc,0x4e);
		rowd = _mm_shuffle_epi32(rowd,0x39);
	// end transpose
	}
	*calc_16 = _mm_add_epi32(*calc_16,rowa);
	*calc_12 = _mm_add_epi32(*calc_12, rowb);
	*calc_8 = _mm_add_epi32(*calc_8, rowc);
	*calc_7 = _mm_add_epi32(*calc_7, rowd);
}

static inline void xor_chacha_sidm_swap(__m128i *calc_16, __m128i *calc_12, __m128i *calc_8, __m128i *calc_7,
 								  __m128i *calc_1, __m128i *calc_2, __m128i *calc_3, __m128i *calc_4,
 								  uint32_t double_rounds)
{
	int i;
	__m128i _calc;
	__m128i _rotate_16 = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
	__m128i _rotate_8  = _mm_setr_epi8(3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14);
	__m128i rowa = _mm_xor_si128(*calc_16, *calc_1);
	__m128i rowb = _mm_xor_si128(*calc_12, *calc_2);
	__m128i rowc = _mm_xor_si128(*calc_8, *calc_3);
	__m128i rowd = _mm_xor_si128(*calc_7, *calc_4);

	*calc_16 = _mm_xor_si128(*calc_16, *calc_1);
	*calc_12 = _mm_xor_si128(*calc_12, *calc_2);
	*calc_8 = _mm_xor_si128(*calc_8, *calc_3);
	*calc_7 = _mm_xor_si128(*calc_7, *calc_4);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd = _mm_shuffle_epi8(rowd,_rotate_16);

		/* second row */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (12));
		_calc = _mm_srli_epi32(_calc,(32 - 12));
		rowb = _mm_xor_si128(rowb, _calc);

		/* third row */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd = _mm_shuffle_epi8(rowd,_rotate_8);

		/* fourth row */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (7));
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		rowb = _mm_xor_si128(rowb, _calc);

// row a		0  1  2  3        0  1  2  3
// row b		4  5  6  7   -->  5  6  7  4
// row c		8  9 10 11   ..> 10 11  8  9
// row d 	   12 13 14 15       15 12 13 14
	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb = _mm_shuffle_epi32(rowb,0x39); // 10 01 00 11
		rowc = _mm_shuffle_epi32(rowc,0x4e); // 01 00 11 10
		rowd = _mm_shuffle_epi32(rowd,0x93); // 00 11 10 01
	// end transpose

		/* first column */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd = _mm_shuffle_epi8(rowd,_rotate_16);

		/* second column */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (12));
		_calc = _mm_srli_epi32(_calc,(32 - 12));
		rowb = _mm_xor_si128(rowb, _calc);

		/* third column */
		rowa = _mm_add_epi32(rowa, rowb);
		rowd = _mm_xor_si128(rowd, rowa);
		rowd = _mm_shuffle_epi8(rowd,_rotate_8);

		/* fourth column */
		rowc = _mm_add_epi32(rowc, rowd);
		_calc = _mm_xor_si128(rowb, rowc);
		rowb = _mm_slli_epi32(_calc, (7));
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		rowb = _mm_xor_si128(rowb, _calc);

		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb = _mm_shuffle_epi32(rowb,0x93);
		rowc = _mm_shuffle_epi32(rowc,0x4e);
		rowd = _mm_shuffle_epi32(rowd,0x39);
	// end transpose
	}

	rowa = _mm_add_epi32(*calc_16,rowa);
	rowb = _mm_add_epi32(*calc_12, rowb);
	rowc = _mm_add_epi32(*calc_8, rowc);
	rowd = _mm_add_epi32(*calc_7, rowd);

	*calc_16 = *calc_1;
	*calc_12 = *calc_2;
	*calc_8 = *calc_3;
	*calc_7 = *calc_4;

	*calc_1 = rowa;
	*calc_2 = rowb;
	*calc_3 = rowc;
	*calc_4 = rowd;
}

static inline void chacha_core_r2_sidm(__m128i *X , uint32_t Loops, uint32_t double_rounds)
{
	uint32_t i, j;
	__m128i scratch[Loops * 8 * 4];

	__m128i *calc_1 = (__m128i*) &X[0];
	__m128i *calc_2 = (__m128i*) &X[1];
	__m128i *calc_3 = (__m128i*) &X[2];
	__m128i *calc_4 = (__m128i*) &X[3];

	__m128i *calc_11 = (__m128i*) &X[4];
	__m128i *calc_12 = (__m128i*) &X[5];
	__m128i *calc_13 = (__m128i*) &X[6];
	__m128i *calc_14 = (__m128i*) &X[7];

	__m128i *calc_21 = (__m128i*) &X[8];
	__m128i *calc_22 = (__m128i*) &X[9];
	__m128i *calc_23 = (__m128i*) &X[10];
	__m128i *calc_24 = (__m128i*) &X[11];

	__m128i *calc_31 = (__m128i*) &X[12];
	__m128i *calc_32 = (__m128i*) &X[13];
	__m128i *calc_33 = (__m128i*) &X[14];
	__m128i *calc_34 = (__m128i*) &X[15];

	for (i = 0; i < Loops; i++) {
		scratch[i * 16 + 0] = *calc_1;		scratch[i * 16 + 1] = *calc_2;
		scratch[i * 16 + 2] = *calc_3;		scratch[i * 16 + 3] = *calc_4;
		scratch[i * 16 + 4] = *calc_11;		scratch[i * 16 + 5] = *calc_12;
		scratch[i * 16 + 6] = *calc_13;		scratch[i * 16 + 7] = *calc_14;
		scratch[i * 16 + 8] = *calc_21;		scratch[i * 16 + 9] = *calc_22;
		scratch[i * 16 + 10] = *calc_23;	scratch[i * 16 + 11] = *calc_24;
		scratch[i * 16 + 12] = *calc_31;	scratch[i * 16 + 13] = *calc_32;
		scratch[i * 16 + 14] = *calc_33;	scratch[i * 16 + 15] = *calc_34;

		xor_chacha_sidm( calc_1, calc_2, calc_3, calc_4, calc_31,calc_32,calc_33,calc_34, double_rounds);
		xor_chacha_sidm(calc_11,calc_12,calc_13,calc_14,  calc_1, calc_2, calc_3, calc_4, double_rounds);
		xor_chacha_sidm_swap(calc_21,calc_22,calc_23,calc_24, calc_11,calc_12,calc_13,calc_14, double_rounds);
		xor_chacha_sidm(calc_31,calc_32,calc_33,calc_34, calc_11,calc_12,calc_13,calc_14, double_rounds);
		// swap calc_2x with calc_1x
	}

	for (i = 0; i < Loops; i++) {
		j = 16 * (_mm_extract_epi16(*calc_31,0x00) & (Loops-1));

		*calc_1 = _mm_xor_si128(*calc_1, scratch[j]);
		*calc_2 = _mm_xor_si128(*calc_2, scratch[j+1]);
		*calc_3 = _mm_xor_si128(*calc_3, scratch[j+2]);
		*calc_4 = _mm_xor_si128(*calc_4, scratch[j+3]);
		*calc_11 = _mm_xor_si128(*calc_11, scratch[j+4]);
		*calc_12 = _mm_xor_si128(*calc_12, scratch[j+5]);
		*calc_13 = _mm_xor_si128(*calc_13, scratch[j+6]);
		*calc_14 = _mm_xor_si128(*calc_14, scratch[j+7]);
		*calc_21 = _mm_xor_si128(*calc_21, scratch[j+8]);
		*calc_22 = _mm_xor_si128(*calc_22, scratch[j+9]);
		*calc_23 = _mm_xor_si128(*calc_23, scratch[j+10]);
		*calc_24 = _mm_xor_si128(*calc_24, scratch[j+11]);
		*calc_31 = _mm_xor_si128(*calc_31, scratch[j+12]);
		*calc_32 = _mm_xor_si128(*calc_32, scratch[j+13]);
		*calc_33 = _mm_xor_si128(*calc_33, scratch[j+14]);
		*calc_34 = _mm_xor_si128(*calc_34, scratch[j+15]);

		xor_chacha_sidm( calc_1, calc_2, calc_3, calc_4, calc_31,calc_32,calc_33,calc_34, double_rounds);
		xor_chacha_sidm(calc_11,calc_12,calc_13,calc_14, calc_1,  calc_2, calc_3, calc_4, double_rounds);
		xor_chacha_sidm_swap(calc_21,calc_22,calc_23,calc_24, calc_11,calc_12,calc_13,calc_14, double_rounds);
		xor_chacha_sidm(calc_31,calc_32,calc_33,calc_34, calc_11,calc_12,calc_13,calc_14, double_rounds);
	}

}

#endif

