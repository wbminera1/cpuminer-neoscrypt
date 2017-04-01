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

#if defined(__x86_64__)

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

//---------------------------------------------------------------------------------------------
//  threefold
//---------------------------------------------------------------------------------------------


static inline void xor_chacha_sidm_X3(
		__m128i *calc_16_1, __m128i *calc_12_1, __m128i *calc_8_1, __m128i *calc_7_1,
 		__m128i *calc_1_1, __m128i *calc_2_1, __m128i *calc_3_1, __m128i *calc_4_1,
		__m128i *calc_16_2, __m128i *calc_12_2, __m128i *calc_8_2, __m128i *calc_7_2,
 		__m128i *calc_1_2, __m128i *calc_2_2, __m128i *calc_3_2, __m128i *calc_4_2,
		__m128i *calc_16_3, __m128i *calc_12_3, __m128i *calc_8_3, __m128i *calc_7_3,
 		__m128i *calc_1_3, __m128i *calc_2_3, __m128i *calc_3_3, __m128i *calc_4_3,
 								  uint32_t double_rounds)
{
	int i;
	__m128i _calc1, _calc2;
	__m128i _rotate_16 = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
	__m128i _rotate_8  = _mm_setr_epi8(3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14);
	__m128i rowa_1 = _mm_xor_si128(*calc_16_1, *calc_1_1);
	__m128i rowb_1 = _mm_xor_si128(*calc_12_1, *calc_2_1);
	__m128i rowc_1 = _mm_xor_si128(*calc_8_1, *calc_3_1);
	__m128i rowd_1 = _mm_xor_si128(*calc_7_1, *calc_4_1);

	__m128i rowa_2 = _mm_xor_si128(*calc_16_2, *calc_1_2);
	__m128i rowb_2 = _mm_xor_si128(*calc_12_2, *calc_2_2);
	__m128i rowc_2 = _mm_xor_si128(*calc_8_2, *calc_3_2);
	__m128i rowd_2 = _mm_xor_si128(*calc_7_2, *calc_4_2);

	__m128i rowa_3 = _mm_xor_si128(*calc_16_3, *calc_1_3);
	__m128i rowb_3 = _mm_xor_si128(*calc_12_3, *calc_2_3);
	__m128i rowc_3 = _mm_xor_si128(*calc_8_3, *calc_3_3);
	__m128i rowd_3 = _mm_xor_si128(*calc_7_3, *calc_4_3);

	*calc_16_1 = _mm_xor_si128(*calc_16_1, *calc_1_1);
	*calc_12_1 = _mm_xor_si128(*calc_12_1, *calc_2_1);
	*calc_8_1 = _mm_xor_si128(*calc_8_1, *calc_3_1);
	*calc_7_1 = _mm_xor_si128(*calc_7_1, *calc_4_1);
	*calc_16_2 = _mm_xor_si128(*calc_16_2, *calc_1_2);
	*calc_12_2 = _mm_xor_si128(*calc_12_2, *calc_2_2);
	*calc_8_2 = _mm_xor_si128(*calc_8_2, *calc_3_2);
	*calc_7_2 = _mm_xor_si128(*calc_7_2, *calc_4_2);
	*calc_16_3 = _mm_xor_si128(*calc_16_3, *calc_1_3);
	*calc_12_3 = _mm_xor_si128(*calc_12_3, *calc_2_3);
	*calc_8_3 = _mm_xor_si128(*calc_8_3, *calc_3_3);
	*calc_7_3 = _mm_xor_si128(*calc_7_3, *calc_4_3);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_16);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_16);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_16);

		/* second row */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (12));
		rowb_3 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 12));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);

		/* third row */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_8);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_8);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_8);

		/* fourth row */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (7));
		rowb_3 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 7));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);

// row a		0  1  2  3        0  1  2  3
// row b		4  5  6  7   -->  5  6  7  4
// row c		8  9 10 11   ..> 10 11  8  9
// row d 	   12 13 14 15       15 12 13 14
	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb_1 = _mm_shuffle_epi32(rowb_1,0x39); // 10 01 00 11
		rowc_1 = _mm_shuffle_epi32(rowc_1,0x4e); // 01 00 11 10
		rowd_1 = _mm_shuffle_epi32(rowd_1,0x93); // 00 11 10 01
		rowb_2 = _mm_shuffle_epi32(rowb_2,0x39); // 10 01 00 11
		rowc_2 = _mm_shuffle_epi32(rowc_2,0x4e); // 01 00 11 10
		rowd_2 = _mm_shuffle_epi32(rowd_2,0x93); // 00 11 10 01
		rowb_3 = _mm_shuffle_epi32(rowb_3,0x39); // 10 01 00 11
		rowc_3 = _mm_shuffle_epi32(rowc_3,0x4e); // 01 00 11 10
		rowd_3 = _mm_shuffle_epi32(rowd_3,0x93); // 00 11 10 01
	// end transpose

		/* first column */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_16);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_16);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_16);

		/* second column */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (12));
		rowb_3 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 12));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);

		/* third column */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_8);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_8);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_8);

		/* fourth column */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (7));
		rowb_3 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 7));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);
		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb_1 = _mm_shuffle_epi32(rowb_1,0x93);
		rowc_1 = _mm_shuffle_epi32(rowc_1,0x4e);
		rowd_1 = _mm_shuffle_epi32(rowd_1,0x39);
		rowb_2 = _mm_shuffle_epi32(rowb_2,0x93);
		rowc_2 = _mm_shuffle_epi32(rowc_2,0x4e);
		rowd_2 = _mm_shuffle_epi32(rowd_2,0x39);
		rowb_3 = _mm_shuffle_epi32(rowb_3,0x93);
		rowc_3 = _mm_shuffle_epi32(rowc_3,0x4e);
		rowd_3 = _mm_shuffle_epi32(rowd_3,0x39);
	// end transpose
	}
	*calc_16_1 = _mm_add_epi32(*calc_16_1, rowa_1);
	*calc_12_1 = _mm_add_epi32(*calc_12_1, rowb_1);
	*calc_8_1 = _mm_add_epi32(  *calc_8_1, rowc_1);
	*calc_7_1 = _mm_add_epi32(  *calc_7_1, rowd_1);
	*calc_16_2 = _mm_add_epi32(*calc_16_2, rowa_2);
	*calc_12_2 = _mm_add_epi32(*calc_12_2, rowb_2);
	*calc_8_2 = _mm_add_epi32(  *calc_8_2, rowc_2);
	*calc_7_2 = _mm_add_epi32(  *calc_7_2, rowd_2);
	*calc_16_3 = _mm_add_epi32(*calc_16_3, rowa_3);
	*calc_12_3 = _mm_add_epi32(*calc_12_3, rowb_3);
	*calc_8_3 = _mm_add_epi32(  *calc_8_3, rowc_3);
	*calc_7_3 = _mm_add_epi32(  *calc_7_3, rowd_3);
}

static inline void xor_chacha_sidm_swap_X3(
		__m128i *calc_16_1, __m128i *calc_12_1, __m128i *calc_8_1, __m128i *calc_7_1,
 		__m128i *calc_1_1, __m128i *calc_2_1, __m128i *calc_3_1, __m128i *calc_4_1,
		__m128i *calc_16_2, __m128i *calc_12_2, __m128i *calc_8_2, __m128i *calc_7_2,
 		__m128i *calc_1_2, __m128i *calc_2_2, __m128i *calc_3_2, __m128i *calc_4_2,
		__m128i *calc_16_3, __m128i *calc_12_3, __m128i *calc_8_3, __m128i *calc_7_3,
 		__m128i *calc_1_3, __m128i *calc_2_3, __m128i *calc_3_3, __m128i *calc_4_3,
 								  uint32_t double_rounds)
{
	int i;
	__m128i _calc1, _calc2;
	__m128i _rotate_16 = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
	__m128i _rotate_8  = _mm_setr_epi8(3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14);
	__m128i rowa_1 = _mm_xor_si128(*calc_16_1, *calc_1_1);
	__m128i rowb_1 = _mm_xor_si128(*calc_12_1, *calc_2_1);
	__m128i rowc_1 = _mm_xor_si128(*calc_8_1, *calc_3_1);
	__m128i rowd_1 = _mm_xor_si128(*calc_7_1, *calc_4_1);

	__m128i rowa_2 = _mm_xor_si128(*calc_16_2, *calc_1_2);
	__m128i rowb_2 = _mm_xor_si128(*calc_12_2, *calc_2_2);
	__m128i rowc_2 = _mm_xor_si128(*calc_8_2, *calc_3_2);
	__m128i rowd_2 = _mm_xor_si128(*calc_7_2, *calc_4_2);

	__m128i rowa_3 = _mm_xor_si128(*calc_16_3, *calc_1_3);
	__m128i rowb_3 = _mm_xor_si128(*calc_12_3, *calc_2_3);
	__m128i rowc_3 = _mm_xor_si128(*calc_8_3, *calc_3_3);
	__m128i rowd_3 = _mm_xor_si128(*calc_7_3, *calc_4_3);

	*calc_16_1 = _mm_xor_si128(*calc_16_1, *calc_1_1);
	*calc_12_1 = _mm_xor_si128(*calc_12_1, *calc_2_1);
	*calc_8_1 = _mm_xor_si128(*calc_8_1, *calc_3_1);
	*calc_7_1 = _mm_xor_si128(*calc_7_1, *calc_4_1);
	*calc_16_2 = _mm_xor_si128(*calc_16_2, *calc_1_2);
	*calc_12_2 = _mm_xor_si128(*calc_12_2, *calc_2_2);
	*calc_8_2 = _mm_xor_si128(*calc_8_2, *calc_3_2);
	*calc_7_2 = _mm_xor_si128(*calc_7_2, *calc_4_2);
	*calc_16_3 = _mm_xor_si128(*calc_16_3, *calc_1_3);
	*calc_12_3 = _mm_xor_si128(*calc_12_3, *calc_2_3);
	*calc_8_3 = _mm_xor_si128(*calc_8_3, *calc_3_3);
	*calc_7_3 = _mm_xor_si128(*calc_7_3, *calc_4_3);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_16);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_16);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_16);

		/* second row */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (12));
		rowb_3 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 12));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);

		/* third row */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_8);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_8);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_8);

		/* fourth row */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (7));
		rowb_3 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 7));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);

// row a		0  1  2  3        0  1  2  3
// row b		4  5  6  7   -->  5  6  7  4
// row c		8  9 10 11   ..> 10 11  8  9
// row d 	   12 13 14 15       15 12 13 14
	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb_1 = _mm_shuffle_epi32(rowb_1,0x39); // 10 01 00 11
		rowc_1 = _mm_shuffle_epi32(rowc_1,0x4e); // 01 00 11 10
		rowd_1 = _mm_shuffle_epi32(rowd_1,0x93); // 00 11 10 01
		rowb_2 = _mm_shuffle_epi32(rowb_2,0x39); // 10 01 00 11
		rowc_2 = _mm_shuffle_epi32(rowc_2,0x4e); // 01 00 11 10
		rowd_2 = _mm_shuffle_epi32(rowd_2,0x93); // 00 11 10 01
		rowb_3 = _mm_shuffle_epi32(rowb_3,0x39); // 10 01 00 11
		rowc_3 = _mm_shuffle_epi32(rowc_3,0x4e); // 01 00 11 10
		rowd_3 = _mm_shuffle_epi32(rowd_3,0x93); // 00 11 10 01
	// end transpose

		/* first column */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_16);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_16);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_16);

		/* second column */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (12));
		rowb_3 = _mm_slli_epi32(_calc1, (12));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 12));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 12));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);

		/* third column */
		rowa_1 = _mm_add_epi32(rowa_1, rowb_1);
		rowa_2 = _mm_add_epi32(rowa_2, rowb_2);
		rowa_3 = _mm_add_epi32(rowa_3, rowb_3);
		rowd_1 = _mm_xor_si128(rowd_1, rowa_1);
		rowd_2 = _mm_xor_si128(rowd_2, rowa_2);
		rowd_3 = _mm_xor_si128(rowd_3, rowa_3);
		rowd_1 =  _mm_shuffle_epi8(rowd_1,_rotate_8);
		rowd_2 =  _mm_shuffle_epi8(rowd_2,_rotate_8);
		rowd_3 =  _mm_shuffle_epi8(rowd_3,_rotate_8);

		/* fourth column */
		rowc_1 = _mm_add_epi32(rowc_1, rowd_1);
		rowc_2 = _mm_add_epi32(rowc_2, rowd_2);
		_calc1 = _mm_xor_si128(rowb_1, rowc_1);
		rowc_3 = _mm_add_epi32(rowc_3, rowd_3);
		rowb_1 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_xor_si128(rowb_2, rowc_2);
		rowb_1 = _mm_xor_si128(rowb_1, _calc1);
		_calc1 = _mm_xor_si128(rowb_3, rowc_3);
		rowb_2 = _mm_slli_epi32(_calc2, (7));
		rowb_3 = _mm_slli_epi32(_calc1, (7));
		_calc1 = _mm_srli_epi32(_calc1,(32 - 7));
		_calc2 = _mm_srli_epi32(_calc2,(32 - 7));
		rowb_2 = _mm_xor_si128(rowb_2, _calc2);
		rowb_3 = _mm_xor_si128(rowb_3, _calc1);
		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		rowb_1 = _mm_shuffle_epi32(rowb_1,0x93);
		rowc_1 = _mm_shuffle_epi32(rowc_1,0x4e);
		rowd_1 = _mm_shuffle_epi32(rowd_1,0x39);
		rowb_2 = _mm_shuffle_epi32(rowb_2,0x93);
		rowc_2 = _mm_shuffle_epi32(rowc_2,0x4e);
		rowd_2 = _mm_shuffle_epi32(rowd_2,0x39);
		rowb_3 = _mm_shuffle_epi32(rowb_3,0x93);
		rowc_3 = _mm_shuffle_epi32(rowc_3,0x4e);
		rowd_3 = _mm_shuffle_epi32(rowd_3,0x39);
	// end transpose
	}
	rowa_1 = _mm_add_epi32(*calc_16_1, rowa_1);
	rowb_1 = _mm_add_epi32(*calc_12_1, rowb_1);
	rowc_1 = _mm_add_epi32(  *calc_8_1, rowc_1);
	rowd_1 = _mm_add_epi32(  *calc_7_1, rowd_1);
	rowa_2 = _mm_add_epi32(*calc_16_2, rowa_2);
	rowb_2 = _mm_add_epi32(*calc_12_2, rowb_2);
	rowc_2 = _mm_add_epi32(  *calc_8_2, rowc_2);
	rowd_2 = _mm_add_epi32(  *calc_7_2, rowd_2);
	rowa_3 = _mm_add_epi32(*calc_16_3, rowa_3);
	rowb_3 = _mm_add_epi32(*calc_12_3, rowb_3);
	rowc_3 = _mm_add_epi32(  *calc_8_3, rowc_3);
	rowd_3 = _mm_add_epi32(  *calc_7_3, rowd_3);

	*calc_16_1 = *calc_1_1;
	*calc_12_1 = *calc_2_1;
	*calc_8_1 = *calc_3_1;
	*calc_7_1 = *calc_4_1;
	*calc_16_2= *calc_1_2;
	*calc_12_2 = *calc_2_2;
	*calc_8_2 = *calc_3_2;
	*calc_7_2 = *calc_4_2;
	*calc_16_3 = *calc_1_3;
	*calc_12_3 = *calc_2_3;
	*calc_8_3 = *calc_3_3;
	*calc_7_3 = *calc_4_3;

	*calc_1_1 = rowa_1;
	*calc_2_1 = rowb_1;
	*calc_3_1 = rowc_1;
	*calc_4_1 = rowd_1;
	*calc_1_2 = rowa_2;
	*calc_2_2 = rowb_2;
	*calc_3_2 = rowc_2;
	*calc_4_2 = rowd_2;
	*calc_1_3 = rowa_3;
	*calc_2_3 = rowb_3;
	*calc_3_3 = rowc_3;
	*calc_4_3 = rowd_3;
}

static inline void chacha_core_r2_sidm_X3(__m128i *X_1, __m128i *X_2, __m128i *X_3, uint32_t Loops, uint32_t double_rounds)
{
	uint32_t i, j1, j2, j3;
	__m128i scratch_1[Loops * 8 * 4];
	__m128i scratch_2[Loops * 8 * 4];
	__m128i scratch_3[Loops * 8 * 4];

	// 1
	__m128i *calc_1_1 = (__m128i*) &X_1[0];
	__m128i *calc_2_1 = (__m128i*) &X_1[1];
	__m128i *calc_3_1 = (__m128i*) &X_1[2];
	__m128i *calc_4_1 = (__m128i*) &X_1[3];

	__m128i *calc_11_1 = (__m128i*) &X_1[4];
	__m128i *calc_12_1 = (__m128i*) &X_1[5];
	__m128i *calc_13_1 = (__m128i*) &X_1[6];
	__m128i *calc_14_1 = (__m128i*) &X_1[7];

	__m128i *calc_21_1 = (__m128i*) &X_1[8];
	__m128i *calc_22_1 = (__m128i*) &X_1[9];
	__m128i *calc_23_1 = (__m128i*) &X_1[10];
	__m128i *calc_24_1 = (__m128i*) &X_1[11];

	__m128i *calc_31_1 = (__m128i*) &X_1[12];
	__m128i *calc_32_1 = (__m128i*) &X_1[13];
	__m128i *calc_33_1 = (__m128i*) &X_1[14];
	__m128i *calc_34_1 = (__m128i*) &X_1[15];
	// 2
	__m128i *calc_1_2 = (__m128i*) &X_2[0];
	__m128i *calc_2_2 = (__m128i*) &X_2[1];
	__m128i *calc_3_2 = (__m128i*) &X_2[2];
	__m128i *calc_4_2 = (__m128i*) &X_2[3];

	__m128i *calc_11_2 = (__m128i*) &X_2[4];
	__m128i *calc_12_2 = (__m128i*) &X_2[5];
	__m128i *calc_13_2 = (__m128i*) &X_2[6];
	__m128i *calc_14_2 = (__m128i*) &X_2[7];

	__m128i *calc_21_2 = (__m128i*) &X_2[8];
	__m128i *calc_22_2 = (__m128i*) &X_2[9];
	__m128i *calc_23_2 = (__m128i*) &X_2[10];
	__m128i *calc_24_2 = (__m128i*) &X_2[11];

	__m128i *calc_31_2 = (__m128i*) &X_2[12];
	__m128i *calc_32_2 = (__m128i*) &X_2[13];
	__m128i *calc_33_2 = (__m128i*) &X_2[14];
	__m128i *calc_34_2 = (__m128i*) &X_2[15];
	// 3
	__m128i *calc_1_3 = (__m128i*) &X_3[0];
	__m128i *calc_2_3 = (__m128i*) &X_3[1];
	__m128i *calc_3_3 = (__m128i*) &X_3[2];
	__m128i *calc_4_3 = (__m128i*) &X_3[3];

	__m128i *calc_11_3 = (__m128i*) &X_3[4];
	__m128i *calc_12_3 = (__m128i*) &X_3[5];
	__m128i *calc_13_3 = (__m128i*) &X_3[6];
	__m128i *calc_14_3 = (__m128i*) &X_3[7];

	__m128i *calc_21_3 = (__m128i*) &X_3[8];
	__m128i *calc_22_3 = (__m128i*) &X_3[9];
	__m128i *calc_23_3 = (__m128i*) &X_3[10];
	__m128i *calc_24_3 = (__m128i*) &X_3[11];

	__m128i *calc_31_3 = (__m128i*) &X_3[12];
	__m128i *calc_32_3 = (__m128i*) &X_3[13];
	__m128i *calc_33_3 = (__m128i*) &X_3[14];
	__m128i *calc_34_3 = (__m128i*) &X_3[15];


	for (i = 0; i < Loops; i++) {
		scratch_1[i * 16 + 0] = *calc_1_1;		scratch_1[i * 16 + 1] = *calc_2_1;
		scratch_1[i * 16 + 2] = *calc_3_1;		scratch_1[i * 16 + 3] = *calc_4_1;
		scratch_2[i * 16 + 0] = *calc_1_2;		scratch_2[i * 16 + 1] = *calc_2_2;
		scratch_2[i * 16 + 2] = *calc_3_2;		scratch_2[i * 16 + 3] = *calc_4_2;
		scratch_3[i * 16 + 0] = *calc_1_3;		scratch_3[i * 16 + 1] = *calc_2_3;
		scratch_3[i * 16 + 2] = *calc_3_3;		scratch_3[i * 16 + 3] = *calc_4_3;

		scratch_1[i * 16 + 12] = *calc_31_1;	scratch_1[i * 16 + 13] = *calc_32_1;
		scratch_1[i * 16 + 14] = *calc_33_1;	scratch_1[i * 16 + 15] = *calc_34_1;
		scratch_2[i * 16 + 12] = *calc_31_2;	scratch_2[i * 16 + 13] = *calc_32_2;
		scratch_2[i * 16 + 14] = *calc_33_2;	scratch_2[i * 16 + 15] = *calc_34_2;
		scratch_3[i * 16 + 12] = *calc_31_3;	scratch_3[i * 16 + 13] = *calc_32_3;
		scratch_3[i * 16 + 14] = *calc_33_3;	scratch_3[i * 16 + 15] = *calc_34_3;

		xor_chacha_sidm_X3( calc_1_1, calc_2_1, calc_3_1, calc_4_1, calc_31_1,calc_32_1,calc_33_1,calc_34_1,
					  	    calc_1_2, calc_2_2, calc_3_2, calc_4_2, calc_31_2,calc_32_2,calc_33_2,calc_34_2,
				            calc_1_3, calc_2_3, calc_3_3, calc_4_3, calc_31_3,calc_32_3,calc_33_3,calc_34_3,
				            double_rounds);

		scratch_1[i * 16 + 4] = *calc_11_1;		scratch_1[i * 16 + 5] = *calc_12_1;
		scratch_1[i * 16 + 6] = *calc_13_1;		scratch_1[i * 16 + 7] = *calc_14_1;
		scratch_2[i * 16 + 4] = *calc_11_2;		scratch_2[i * 16 + 5] = *calc_12_2;
		scratch_2[i * 16 + 6] = *calc_13_2;		scratch_2[i * 16 + 7] = *calc_14_2;
		scratch_3[i * 16 + 4] = *calc_11_3;		scratch_3[i * 16 + 5] = *calc_12_3;
		scratch_3[i * 16 + 6] = *calc_13_3;		scratch_3[i * 16 + 7] = *calc_14_3;

		xor_chacha_sidm_X3( calc_11_1, calc_12_1, calc_13_1, calc_14_1, calc_1_1,calc_2_1,calc_3_1,calc_4_1,
						    calc_11_2, calc_12_2, calc_13_2, calc_14_2, calc_1_2,calc_2_2,calc_3_2,calc_4_2,
						    calc_11_3, calc_12_3, calc_13_3, calc_14_3, calc_1_3,calc_2_3,calc_3_3,calc_4_3,
						    double_rounds);

		scratch_1[i * 16 +  8] = *calc_21_1;	scratch_1[i * 16 +  9] = *calc_22_1;
		scratch_1[i * 16 + 10] = *calc_23_1;	scratch_1[i * 16 + 11] = *calc_24_1;
		scratch_2[i * 16 +  8] = *calc_21_2;	scratch_2[i * 16 +  9] = *calc_22_2;
		scratch_2[i * 16 + 10] = *calc_23_2;	scratch_2[i * 16 + 11] = *calc_24_2;
		scratch_3[i * 16 +  8] = *calc_21_3;	scratch_3[i * 16 +  9] = *calc_22_3;
		scratch_3[i * 16 + 10] = *calc_23_3;	scratch_3[i * 16 + 11] = *calc_24_3;

		xor_chacha_sidm_swap_X3(  calc_21_1, calc_22_1, calc_23_1, calc_24_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
 							      calc_21_2, calc_22_2, calc_23_2, calc_24_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							      calc_21_3, calc_22_3, calc_23_3, calc_24_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							      double_rounds);

		xor_chacha_sidm_X3(  calc_31_1, calc_32_1, calc_33_1, calc_34_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
							 calc_31_2, calc_32_2, calc_33_2, calc_34_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							 calc_31_3, calc_32_3, calc_33_3, calc_34_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							 double_rounds);		// swap calc_2x with calc_1x
	}

	for (i = 0; i < Loops; i++) {
		j1 = 16 * (_mm_extract_epi16(*calc_31_1,0x00) & (Loops-1));
		j2 = 16 * (_mm_extract_epi16(*calc_31_2,0x00) & (Loops-1));
		j3 = 16 * (_mm_extract_epi16(*calc_31_3,0x00) & (Loops-1));

		//1
		*calc_1_1 = _mm_xor_si128(*calc_1_1, scratch_1[j1]);
		*calc_2_1 = _mm_xor_si128(*calc_2_1, scratch_1[j1+1]);
		*calc_3_1 = _mm_xor_si128(*calc_3_1, scratch_1[j1+2]);
		*calc_4_1 = _mm_xor_si128(*calc_4_1, scratch_1[j1+3]);

		*calc_31_1 = _mm_xor_si128(*calc_31_1, scratch_1[j1+12]);
		*calc_32_1 = _mm_xor_si128(*calc_32_1, scratch_1[j1+13]);
		*calc_33_1 = _mm_xor_si128(*calc_33_1, scratch_1[j1+14]);
		*calc_34_1 = _mm_xor_si128(*calc_34_1, scratch_1[j1+15]);
		//2
		*calc_1_2 = _mm_xor_si128(*calc_1_2, scratch_2[j2]);
		*calc_2_2 = _mm_xor_si128(*calc_2_2, scratch_2[j2+1]);
		*calc_3_2 = _mm_xor_si128(*calc_3_2, scratch_2[j2+2]);
		*calc_4_2 = _mm_xor_si128(*calc_4_2, scratch_2[j2+3]);

		*calc_31_2 = _mm_xor_si128(*calc_31_2, scratch_2[j2+12]);
		*calc_32_2 = _mm_xor_si128(*calc_32_2, scratch_2[j2+13]);
		*calc_33_2 = _mm_xor_si128(*calc_33_2, scratch_2[j2+14]);
		*calc_34_2 = _mm_xor_si128(*calc_34_2, scratch_2[j2+15]);
		//3
		*calc_1_3 = _mm_xor_si128(*calc_1_3, scratch_3[j3]);
		*calc_2_3 = _mm_xor_si128(*calc_2_3, scratch_3[j3+1]);
		*calc_3_3 = _mm_xor_si128(*calc_3_3, scratch_3[j3+2]);
		*calc_4_3 = _mm_xor_si128(*calc_4_3, scratch_3[j3+3]);

		*calc_31_3 = _mm_xor_si128(*calc_31_3, scratch_3[j3+12]);
		*calc_32_3 = _mm_xor_si128(*calc_32_3, scratch_3[j3+13]);
		*calc_33_3 = _mm_xor_si128(*calc_33_3, scratch_3[j3+14]);
		*calc_34_3 = _mm_xor_si128(*calc_34_3, scratch_3[j3+15]);

		xor_chacha_sidm_X3( calc_1_1, calc_2_1, calc_3_1, calc_4_1, calc_31_1,calc_32_1,calc_33_1,calc_34_1,
					  	    calc_1_2, calc_2_2, calc_3_2, calc_4_2, calc_31_2,calc_32_2,calc_33_2,calc_34_2,
				            calc_1_3, calc_2_3, calc_3_3, calc_4_3, calc_31_3,calc_32_3,calc_33_3,calc_34_3,
				            double_rounds);

		//1
		*calc_11_1 = _mm_xor_si128(*calc_11_1, scratch_1[j1+4]);
		*calc_12_1 = _mm_xor_si128(*calc_12_1, scratch_1[j1+5]);
		*calc_13_1 = _mm_xor_si128(*calc_13_1, scratch_1[j1+6]);
		*calc_14_1 = _mm_xor_si128(*calc_14_1, scratch_1[j1+7]);
		//2
		*calc_11_2 = _mm_xor_si128(*calc_11_2, scratch_2[j2+4]);
		*calc_12_2 = _mm_xor_si128(*calc_12_2, scratch_2[j2+5]);
		*calc_13_2 = _mm_xor_si128(*calc_13_2, scratch_2[j2+6]);
		*calc_14_2 = _mm_xor_si128(*calc_14_2, scratch_2[j2+7]);
		//3
		*calc_11_3 = _mm_xor_si128(*calc_11_3, scratch_3[j3+4]);
		*calc_12_3 = _mm_xor_si128(*calc_12_3, scratch_3[j3+5]);
		*calc_13_3 = _mm_xor_si128(*calc_13_3, scratch_3[j3+6]);
		*calc_14_3 = _mm_xor_si128(*calc_14_3, scratch_3[j3+7]);

		xor_chacha_sidm_X3( calc_11_1, calc_12_1, calc_13_1, calc_14_1, calc_1_1,calc_2_1,calc_3_1,calc_4_1,
						    calc_11_2, calc_12_2, calc_13_2, calc_14_2, calc_1_2,calc_2_2,calc_3_2,calc_4_2,
						    calc_11_3, calc_12_3, calc_13_3, calc_14_3, calc_1_3,calc_2_3,calc_3_3,calc_4_3,
						    double_rounds);

		//1
		*calc_21_1 = _mm_xor_si128(*calc_21_1, scratch_1[j1+8]);
		*calc_22_1 = _mm_xor_si128(*calc_22_1, scratch_1[j1+9]);
		*calc_23_1 = _mm_xor_si128(*calc_23_1, scratch_1[j1+10]);
		*calc_24_1 = _mm_xor_si128(*calc_24_1, scratch_1[j1+11]);
		//2
		*calc_21_2 = _mm_xor_si128(*calc_21_2, scratch_2[j2+8]);
		*calc_22_2 = _mm_xor_si128(*calc_22_2, scratch_2[j2+9]);
		*calc_23_2 = _mm_xor_si128(*calc_23_2, scratch_2[j2+10]);
		*calc_24_2 = _mm_xor_si128(*calc_24_2, scratch_2[j2+11]);
		//3
		*calc_21_3 = _mm_xor_si128(*calc_21_3, scratch_3[j3+8]);
		*calc_22_3 = _mm_xor_si128(*calc_22_3, scratch_3[j3+9]);
		*calc_23_3 = _mm_xor_si128(*calc_23_3, scratch_3[j3+10]);
		*calc_24_3 = _mm_xor_si128(*calc_24_3, scratch_3[j3+11]);

		xor_chacha_sidm_swap_X3(  calc_21_1, calc_22_1, calc_23_1, calc_24_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
 							      calc_21_2, calc_22_2, calc_23_2, calc_24_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							      calc_21_3, calc_22_3, calc_23_3, calc_24_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							      double_rounds);

		xor_chacha_sidm_X3(  calc_31_1, calc_32_1, calc_33_1, calc_34_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
							 calc_31_2, calc_32_2, calc_33_2, calc_34_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							 calc_31_3, calc_32_3, calc_33_3, calc_34_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							 double_rounds);		// swap calc_2x with calc_1x
	}

}
//---------------------------------------------------------------------------------------------
// end threefold
//---------------------------------------------------------------------------------------------




static inline void xor_sidm(__m128i *dest, __m128i *src, uint32_t size)
{
	uint32_t i;
	for (i=0; i< size; i++){
		dest[i] = _mm_xor_si128(dest[i], src[i]);
	}
}

#endif

