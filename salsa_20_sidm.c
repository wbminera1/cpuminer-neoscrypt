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

static inline void xor_salsa_sidm(__m128i *calc_18, __m128i *calc_13, __m128i *calc_9, __m128i *calc_7,
 								  __m128i *calc_1, __m128i *calc_4, __m128i *calc_3, __m128i *calc_2,
 								  uint32_t double_rounds)
{
	int i;
	__m128i _calc;
	__m128i _shift_left;
	__m128i row1 = _mm_xor_si128(*calc_18, *calc_1);;
	__m128i row2 = _mm_xor_si128(*calc_7, *calc_2);;
	__m128i row3 = _mm_xor_si128(*calc_9, *calc_3);;
	__m128i row4 = _mm_xor_si128(*calc_13, *calc_4);;

	*calc_18 = _mm_xor_si128(*calc_18, *calc_1);
	*calc_7 = _mm_xor_si128(*calc_7, *calc_2);
	*calc_9 = _mm_xor_si128(*calc_9, *calc_3);
	*calc_13 = _mm_xor_si128(*calc_13, *calc_4);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
 		_calc = _mm_add_epi32(row1, row4);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row2 = _mm_xor_si128(row2, _shift_left);
		row2 = _mm_xor_si128(row2, _calc);

		/* second row */
		_calc = _mm_add_epi32(row2, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _shift_left);
		row3 = _mm_xor_si128(row3, _calc);

		/* third row */
		_calc = _mm_add_epi32(row3, row2);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row4 = _mm_xor_si128(row4, _shift_left);
		row4 = _mm_xor_si128(row4, _calc);

		/* fourth row */
		_calc = _mm_add_epi32(row4, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _shift_left);
		row1 = _mm_xor_si128(row1, _calc);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x93);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x39);
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column */
		_calc = _mm_add_epi32(row1, row2);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row4 = _mm_xor_si128(row4, _shift_left);
		row4 = _mm_xor_si128(row4, _calc);

		/* second column */
		_calc = _mm_add_epi32(row4, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _shift_left);
		row3 = _mm_xor_si128(row3, _calc);

		/* third column */
		_calc = _mm_add_epi32(row3, row4);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row2 = _mm_xor_si128(row2, _shift_left);
		row2 = _mm_xor_si128(row2, _calc);

		/* fourth column */
		_calc = _mm_add_epi32(row2, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _shift_left);
		row1 = _mm_xor_si128(row1, _calc);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x39);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x93);
	// end transpose
	}
	*calc_18 = _mm_add_epi32(*calc_18,row1);
	*calc_7 = _mm_add_epi32(*calc_7, row2);
	*calc_9 = _mm_add_epi32(*calc_9, row3);
	*calc_13 = _mm_add_epi32(*calc_13, row4);
}

static inline void xor_salsa_sidm_swap(__m128i *calc_18, __m128i *calc_13, __m128i *calc_9, __m128i *calc_7,
 								  __m128i *calc_1, __m128i *calc_4, __m128i *calc_3, __m128i *calc_2,
 								  uint32_t double_rounds)
{
	int i;
	__m128i _calc;
	__m128i _shift_left;
	__m128i row1 = _mm_xor_si128(*calc_18, *calc_1);;
	__m128i row2 = _mm_xor_si128(*calc_7, *calc_2);;
	__m128i row3 = _mm_xor_si128(*calc_9, *calc_3);;
	__m128i row4 = _mm_xor_si128(*calc_13, *calc_4);;

	*calc_18 = _mm_xor_si128(*calc_18, *calc_1);
	*calc_7 = _mm_xor_si128(*calc_7, *calc_2);
	*calc_9 = _mm_xor_si128(*calc_9, *calc_3);
	*calc_13 = _mm_xor_si128(*calc_13, *calc_4);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
 		_calc = _mm_add_epi32(row1, row4);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row2 = _mm_xor_si128(row2, _calc);
		row2 = _mm_xor_si128(row2, _shift_left);

		/* second row */
		_calc = _mm_add_epi32(row2, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _calc);
		row3 = _mm_xor_si128(row3, _shift_left);

		/* third row */
		_calc = _mm_add_epi32(row3, row2);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row4 = _mm_xor_si128(row4, _calc);
		row4 = _mm_xor_si128(row4, _shift_left);

		/* fourth row */
		_calc = _mm_add_epi32(row4, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _calc);
		row1 = _mm_xor_si128(row1, _shift_left);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x93);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x39);
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column */
		_calc = _mm_add_epi32(row1, row2);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row4 = _mm_xor_si128(row4, _calc);
		row4 = _mm_xor_si128(row4, _shift_left);

		/* second column */
		_calc = _mm_add_epi32(row4, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _calc);
		row3 = _mm_xor_si128(row3, _shift_left);

		/* third column */
		_calc = _mm_add_epi32(row3, row4);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row2 = _mm_xor_si128(row2, _calc);
		row2 = _mm_xor_si128(row2, _shift_left);

		/* fourth column */
		_calc = _mm_add_epi32(row2, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _calc);
		row1 = _mm_xor_si128(row1, _shift_left);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x39);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x93);
	// end transpose
	}

	row1 = _mm_add_epi32(*calc_18,row1);
	row2 = _mm_add_epi32(*calc_7, row2);
	row3 = _mm_add_epi32(*calc_9, row3);
	row4 = _mm_add_epi32(*calc_13, row4);

	*calc_18 = *calc_1;
	*calc_7 = *calc_2;
	*calc_9 = *calc_3;
	*calc_13 = *calc_4;

	*calc_1 = row1;
	*calc_2 = row2;
	*calc_3 = row3;
	*calc_4 = row4;
}

static inline void scrypt_core_r2_sidm(__m128i *X , uint32_t Loops, uint32_t double_rounds)
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

	__m128i _calc5;
	__m128i _calc6;
	__m128i _calc7;
	__m128i _calc8;

	/* transpose the data from *X */
	_calc5 =_mm_blend_epi16(*calc_31, *calc_33, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32, *calc_34, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33, *calc_31, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34, *calc_32, 0x0f);
	*calc_31 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21, *calc_23, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22, *calc_24, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23, *calc_21, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24, *calc_22, 0x0f);
	*calc_21 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11, *calc_13, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12, *calc_14, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13, *calc_11, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14, *calc_12, 0x0f);
	*calc_11 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1, *calc_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2, *calc_4, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3, *calc_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4, *calc_2, 0x0f);
	*calc_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	for (i = 0; i < Loops; i++) {
		scratch[i * 16 + 0] = *calc_1;		scratch[i * 16 + 1] = *calc_2;
		scratch[i * 16 + 2] = *calc_3;		scratch[i * 16 + 3] = *calc_4;
		scratch[i * 16 + 4] = *calc_11;		scratch[i * 16 + 5] = *calc_12;
		scratch[i * 16 + 6] = *calc_13;		scratch[i * 16 + 7] = *calc_14;
		scratch[i * 16 + 8] = *calc_21;		scratch[i * 16 + 9] = *calc_22;
		scratch[i * 16 + 10] = *calc_23;	scratch[i * 16 + 11] = *calc_24;
		scratch[i * 16 + 12] = *calc_31;	scratch[i * 16 + 13] = *calc_32;
		scratch[i * 16 + 14] = *calc_33;	scratch[i * 16 + 15] = *calc_34;

		xor_salsa_sidm( calc_1, calc_2, calc_3, calc_4,calc_31,calc_32,calc_33,calc_34, double_rounds);
		xor_salsa_sidm(calc_11,calc_12,calc_13,calc_14, calc_1, calc_2, calc_3, calc_4, double_rounds);
		xor_salsa_sidm_swap(calc_21,calc_22,calc_23,calc_24, calc_11, calc_12, calc_13, calc_14, double_rounds);
		xor_salsa_sidm(calc_31,calc_32,calc_33,calc_34, calc_11, calc_12, calc_13, calc_14, double_rounds);
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

		xor_salsa_sidm( calc_1, calc_2, calc_3, calc_4,calc_31,calc_32,calc_33,calc_34, double_rounds);
		xor_salsa_sidm(calc_11,calc_12,calc_13,calc_14, calc_1, calc_2, calc_3, calc_4, double_rounds);
		xor_salsa_sidm_swap(calc_21,calc_22,calc_23,calc_24, calc_11, calc_12, calc_13, calc_14, double_rounds);
		xor_salsa_sidm(calc_31,calc_32,calc_33,calc_34, calc_11, calc_12, calc_13, calc_14, double_rounds);
	}

	_calc5 =_mm_blend_epi16(*calc_31, *calc_33, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32, *calc_34, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33, *calc_31, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34, *calc_32, 0x0f);
	*calc_31 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21, *calc_23, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22, *calc_24, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23, *calc_21, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24, *calc_22, 0x0f);
	*calc_21 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11, *calc_13, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12, *calc_14, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13, *calc_11, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14, *calc_12, 0x0f);
	*calc_11 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1, *calc_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2, *calc_4, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3, *calc_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4, *calc_2, 0x0f);
	*calc_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4 = _mm_blend_epi16(_calc8, _calc7, 0xcc);
}

//---------------------------------------------------------------------------------------------
//  threefold
//---------------------------------------------------------------------------------------------
static inline void xor_salsa_sidm_X3(
		__m128i *calc_18_1, __m128i *calc_13_1, __m128i *calc_9_1, __m128i *calc_7_1,
		__m128i *calc_1_1, __m128i *calc_4_1, __m128i *calc_3_1, __m128i *calc_2_1,
		__m128i *calc_18_2, __m128i *calc_13_2, __m128i *calc_9_2, __m128i *calc_7_2,
		__m128i *calc_1_2, __m128i *calc_4_2, __m128i *calc_3_2, __m128i *calc_2_2,
		__m128i *calc_18_3, __m128i *calc_13_3, __m128i *calc_9_3, __m128i *calc_7_3,
		__m128i *calc_1_3, __m128i *calc_4_3, __m128i *calc_3_3, __m128i *calc_2_3,
 		uint32_t double_rounds)
{
	int i;
	__m128i _calc_1, _calc_2, _calc_3;
	__m128i _shift_left;
	__m128i row1_1 = _mm_xor_si128(*calc_18_1, *calc_1_1);
	__m128i row2_1 = _mm_xor_si128(*calc_7_1, *calc_2_1);
	__m128i row3_1 = _mm_xor_si128(*calc_9_1, *calc_3_1);
	__m128i row4_1 = _mm_xor_si128(*calc_13_1, *calc_4_1);
	__m128i row1_2 = _mm_xor_si128(*calc_18_2, *calc_1_2);
	__m128i row2_2 = _mm_xor_si128(*calc_7_2, *calc_2_2);
	__m128i row3_2 = _mm_xor_si128(*calc_9_2, *calc_3_2);
	__m128i row4_2 = _mm_xor_si128(*calc_13_2, *calc_4_2);
	__m128i row1_3 = _mm_xor_si128(*calc_18_3, *calc_1_3);
	__m128i row2_3 = _mm_xor_si128(*calc_7_3, *calc_2_3);
	__m128i row3_3 = _mm_xor_si128(*calc_9_3, *calc_3_3);
	__m128i row4_3 = _mm_xor_si128(*calc_13_3, *calc_4_3);


	*calc_18_1 = _mm_xor_si128(*calc_18_1, *calc_1_1);
	*calc_7_1 = _mm_xor_si128(*calc_7_1, *calc_2_1);
	*calc_9_1 = _mm_xor_si128(*calc_9_1, *calc_3_1);
	*calc_13_1 = _mm_xor_si128(*calc_13_1, *calc_4_1);

	*calc_18_2 = _mm_xor_si128(*calc_18_2, *calc_1_2);
	*calc_7_2 = _mm_xor_si128(*calc_7_2, *calc_2_2);
	*calc_9_2 = _mm_xor_si128(*calc_9_2, *calc_3_2);
	*calc_13_2 = _mm_xor_si128(*calc_13_2, *calc_4_2);

	*calc_18_3 = _mm_xor_si128(*calc_18_3, *calc_1_3);
	*calc_7_3 = _mm_xor_si128(*calc_7_3, *calc_2_3);
	*calc_9_3 = _mm_xor_si128(*calc_9_3, *calc_3_3);
	*calc_13_3 = _mm_xor_si128(*calc_13_3, *calc_4_3);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
 		_calc_1 = _mm_add_epi32(row1_1, row4_1);
 		_calc_2 = _mm_add_epi32(row1_2, row4_2);
		_shift_left = _mm_slli_epi32(_calc_1, 7);
 		_calc_3 = _mm_add_epi32(row1_3, row4_3);
		row2_1 = _mm_xor_si128(row2_1, _shift_left);
 		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_2, 7);
		row2_1 = _mm_xor_si128(row2_1, _calc_1);
		row2_2 = _mm_xor_si128(row2_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_3, 7);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 7));
		row2_3 = _mm_xor_si128(row2_3, _shift_left);
		row2_2 = _mm_xor_si128(row2_2, _calc_2);
		row2_3 = _mm_xor_si128(row2_3, _calc_3);

		/* second row */
		_calc_1 = _mm_add_epi32(row2_1, row1_1);
		_calc_2 = _mm_add_epi32(row2_2, row1_2);
		_shift_left = _mm_slli_epi32(_calc_1, 9);
		_calc_3 = _mm_add_epi32(row2_3, row1_3);
		row3_1 = _mm_xor_si128(row3_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_2, 9);
		row3_1 = _mm_xor_si128(row3_1, _calc_1);
		row3_2 = _mm_xor_si128(row3_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_3, 9);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 9));
		row3_3 = _mm_xor_si128(row3_3, _shift_left);
		row3_2 = _mm_xor_si128(row3_2, _calc_2);
		row3_3 = _mm_xor_si128(row3_3, _calc_3);

		/* third row */
		_calc_1 = _mm_add_epi32(row3_1, row2_1);
		_calc_2 = _mm_add_epi32(row3_2, row2_2);
		_shift_left = _mm_slli_epi32(_calc_1, 13);
		_calc_3 = _mm_add_epi32(row3_3, row2_3);
		row4_1 = _mm_xor_si128(row4_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_2, 13);
		row4_1 = _mm_xor_si128(row4_1, _calc_1);
		row4_2 = _mm_xor_si128(row4_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_3, 13);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 13));
		row4_3 = _mm_xor_si128(row4_3, _shift_left);
		row4_2 = _mm_xor_si128(row4_2, _calc_2);
		row4_3 = _mm_xor_si128(row4_3, _calc_3);

		/* fourth row */
		_calc_1 = _mm_add_epi32(row4_1, row3_1);
		_calc_2 = _mm_add_epi32(row4_2, row3_2);
		_shift_left = _mm_slli_epi32(_calc_1, 18);
		_calc_3 = _mm_add_epi32(row4_3, row3_3);
		row1_1 = _mm_xor_si128(row1_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_2, 18);
		row1_1 = _mm_xor_si128(row1_1, _calc_1);
		row1_2 = _mm_xor_si128(row1_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_3, 18);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 18));
		row1_3 = _mm_xor_si128(row1_3, _shift_left);
		row1_2 = _mm_xor_si128(row1_2, _calc_2);
		row1_3 = _mm_xor_si128(row1_3, _calc_3);

		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2_1 = _mm_shuffle_epi32(row2_1,0x93);
		row3_1 = _mm_shuffle_epi32(row3_1,0x4e);
		row4_1 = _mm_shuffle_epi32(row4_1,0x39);
		row2_2 = _mm_shuffle_epi32(row2_2,0x93);
		row3_2 = _mm_shuffle_epi32(row3_2,0x4e);
		row4_2 = _mm_shuffle_epi32(row4_2,0x39);
		row2_3 = _mm_shuffle_epi32(row2_3,0x93);
		row3_3 = _mm_shuffle_epi32(row3_3,0x4e);
		row4_3 = _mm_shuffle_epi32(row4_3,0x39);
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column */
		_calc_1 = _mm_add_epi32(row1_1, row2_1);
		_calc_2 = _mm_add_epi32(row1_2, row2_2);
		_shift_left = _mm_slli_epi32(_calc_1, 7);
		_calc_3 = _mm_add_epi32(row1_3, row2_3);
		row4_1 = _mm_xor_si128(row4_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_2, 7);
		row4_1 = _mm_xor_si128(row4_1, _calc_1);
		row4_2 = _mm_xor_si128(row4_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_3, 7);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 7));
		row4_3 = _mm_xor_si128(row4_3, _shift_left);
		row4_2 = _mm_xor_si128(row4_2, _calc_2);
		row4_3 = _mm_xor_si128(row4_3, _calc_3);

		/* second column */
		_calc_1 = _mm_add_epi32(row4_1, row1_1);
		_calc_2 = _mm_add_epi32(row4_2, row1_2);
		_shift_left = _mm_slli_epi32(_calc_1, 9);
		_calc_3 = _mm_add_epi32(row4_3, row1_3);
		row3_1 = _mm_xor_si128(row3_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_2, 9);
		row3_1 = _mm_xor_si128(row3_1, _calc_1);
		row3_2 = _mm_xor_si128(row3_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_3, 9);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 9));
		row3_3 = _mm_xor_si128(row3_3, _shift_left);
		row3_2 = _mm_xor_si128(row3_2, _calc_2);
		row3_3 = _mm_xor_si128(row3_3, _calc_3);

		/* third column */
		_calc_1 = _mm_add_epi32(row3_1, row4_1);
		_calc_2 = _mm_add_epi32(row3_2, row4_2);
		_shift_left = _mm_slli_epi32(_calc_1, 13);
		_calc_3 = _mm_add_epi32(row3_3, row4_3);
		row2_1 = _mm_xor_si128(row2_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_2, 13);
		row2_1 = _mm_xor_si128(row2_1, _calc_1);
		row2_2 = _mm_xor_si128(row2_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_3, 13);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 13));
		row2_3 = _mm_xor_si128(row2_3, _shift_left);
		row2_2 = _mm_xor_si128(row2_2, _calc_2);
		row2_3 = _mm_xor_si128(row2_3, _calc_3);

		/* fourth column */
		_calc_1 = _mm_add_epi32(row2_1, row3_1);
		_calc_2 = _mm_add_epi32(row2_2, row3_2);
		_shift_left = _mm_slli_epi32(_calc_1, 18);
		_calc_3 = _mm_add_epi32(row2_3, row3_3);
		row1_1 = _mm_xor_si128(row1_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_2, 18);
		row1_1 = _mm_xor_si128(row1_1, _calc_1);
		row1_2 = _mm_xor_si128(row1_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_3, 18);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 18));
		row1_3 = _mm_xor_si128(row1_3, _shift_left);
		row1_2 = _mm_xor_si128(row1_2, _calc_2);
		row1_3 = _mm_xor_si128(row1_3, _calc_3);

		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2_1 = _mm_shuffle_epi32(row2_1,0x39);
		row3_1 = _mm_shuffle_epi32(row3_1,0x4e);
		row4_1 = _mm_shuffle_epi32(row4_1,0x93);
		row2_2 = _mm_shuffle_epi32(row2_2,0x39);
		row3_2 = _mm_shuffle_epi32(row3_2,0x4e);
		row4_2 = _mm_shuffle_epi32(row4_2,0x93);
		row2_3 = _mm_shuffle_epi32(row2_3,0x39);
		row3_3 = _mm_shuffle_epi32(row3_3,0x4e);
		row4_3 = _mm_shuffle_epi32(row4_3,0x93);
	// end transpose
	}
	*calc_18_1 = _mm_add_epi32(*calc_18_1,row1_1);
	*calc_7_1 = _mm_add_epi32(*calc_7_1, row2_1);
	*calc_9_1 = _mm_add_epi32(*calc_9_1, row3_1);
	*calc_13_1 = _mm_add_epi32(*calc_13_1, row4_1);

	*calc_18_2 = _mm_add_epi32(*calc_18_2,row1_2);
	*calc_7_2 = _mm_add_epi32(*calc_7_2, row2_2);
	*calc_9_2 = _mm_add_epi32(*calc_9_2, row3_2);
	*calc_13_2 = _mm_add_epi32(*calc_13_2, row4_2);

	*calc_18_3 = _mm_add_epi32(*calc_18_3,row1_3);
	*calc_7_3 = _mm_add_epi32(*calc_7_3, row2_3);
	*calc_9_3 = _mm_add_epi32(*calc_9_3, row3_3);
	*calc_13_3 = _mm_add_epi32(*calc_13_3, row4_3);
}

static inline void xor_salsa_sidm_swap_X3(
		__m128i *calc_18_1, __m128i *calc_13_1, __m128i *calc_9_1, __m128i *calc_7_1,
		__m128i *calc_1_1, __m128i *calc_4_1, __m128i *calc_3_1, __m128i *calc_2_1,
		__m128i *calc_18_2, __m128i *calc_13_2, __m128i *calc_9_2, __m128i *calc_7_2,
		__m128i *calc_1_2, __m128i *calc_4_2, __m128i *calc_3_2, __m128i *calc_2_2,
		__m128i *calc_18_3, __m128i *calc_13_3, __m128i *calc_9_3, __m128i *calc_7_3,
		__m128i *calc_1_3, __m128i *calc_4_3, __m128i *calc_3_3, __m128i *calc_2_3,
 		uint32_t double_rounds)
{
	int i;
	__m128i _calc_1, _calc_2, _calc_3;
	__m128i _shift_left;
	__m128i row1_1 = _mm_xor_si128(*calc_18_1, *calc_1_1);
	__m128i row2_1 = _mm_xor_si128(*calc_7_1, *calc_2_1);
	__m128i row3_1 = _mm_xor_si128(*calc_9_1, *calc_3_1);
	__m128i row4_1 = _mm_xor_si128(*calc_13_1, *calc_4_1);
	__m128i row1_2 = _mm_xor_si128(*calc_18_2, *calc_1_2);
	__m128i row2_2 = _mm_xor_si128(*calc_7_2, *calc_2_2);
	__m128i row3_2 = _mm_xor_si128(*calc_9_2, *calc_3_2);
	__m128i row4_2 = _mm_xor_si128(*calc_13_2, *calc_4_2);
	__m128i row1_3 = _mm_xor_si128(*calc_18_3, *calc_1_3);
	__m128i row2_3 = _mm_xor_si128(*calc_7_3, *calc_2_3);
	__m128i row3_3 = _mm_xor_si128(*calc_9_3, *calc_3_3);
	__m128i row4_3 = _mm_xor_si128(*calc_13_3, *calc_4_3);


	*calc_18_1 = _mm_xor_si128(*calc_18_1, *calc_1_1);
	*calc_7_1 = _mm_xor_si128(*calc_7_1, *calc_2_1);
	*calc_9_1 = _mm_xor_si128(*calc_9_1, *calc_3_1);
	*calc_13_1 = _mm_xor_si128(*calc_13_1, *calc_4_1);

	*calc_18_2 = _mm_xor_si128(*calc_18_2, *calc_1_2);
	*calc_7_2 = _mm_xor_si128(*calc_7_2, *calc_2_2);
	*calc_9_2 = _mm_xor_si128(*calc_9_2, *calc_3_2);
	*calc_13_2 = _mm_xor_si128(*calc_13_2, *calc_4_2);

	*calc_18_3 = _mm_xor_si128(*calc_18_3, *calc_1_3);
	*calc_7_3 = _mm_xor_si128(*calc_7_3, *calc_2_3);
	*calc_9_3 = _mm_xor_si128(*calc_9_3, *calc_3_3);
	*calc_13_3 = _mm_xor_si128(*calc_13_3, *calc_4_3);

	for (i = 0; i < double_rounds; i++) {
		/* first row */
 		_calc_1 = _mm_add_epi32(row1_1, row4_1);
 		_calc_2 = _mm_add_epi32(row1_2, row4_2);
		_shift_left = _mm_slli_epi32(_calc_1, 7);
 		_calc_3 = _mm_add_epi32(row1_3, row4_3);
		row2_1 = _mm_xor_si128(row2_1, _shift_left);
 		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_2, 7);
		row2_1 = _mm_xor_si128(row2_1, _calc_1);
		row2_2 = _mm_xor_si128(row2_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_3, 7);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 7));
		row2_3 = _mm_xor_si128(row2_3, _shift_left);
		row2_2 = _mm_xor_si128(row2_2, _calc_2);
		row2_3 = _mm_xor_si128(row2_3, _calc_3);

		/* second row */
		_calc_1 = _mm_add_epi32(row2_1, row1_1);
		_calc_2 = _mm_add_epi32(row2_2, row1_2);
		_shift_left = _mm_slli_epi32(_calc_1, 9);
		_calc_3 = _mm_add_epi32(row2_3, row1_3);
		row3_1 = _mm_xor_si128(row3_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_2, 9);
		row3_1 = _mm_xor_si128(row3_1, _calc_1);
		row3_2 = _mm_xor_si128(row3_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_3, 9);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 9));
		row3_3 = _mm_xor_si128(row3_3, _shift_left);
		row3_2 = _mm_xor_si128(row3_2, _calc_2);
		row3_3 = _mm_xor_si128(row3_3, _calc_3);

		/* third row */
		_calc_1 = _mm_add_epi32(row3_1, row2_1);
		_calc_2 = _mm_add_epi32(row3_2, row2_2);
		_shift_left = _mm_slli_epi32(_calc_1, 13);
		_calc_3 = _mm_add_epi32(row3_3, row2_3);
		row4_1 = _mm_xor_si128(row4_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_2, 13);
		row4_1 = _mm_xor_si128(row4_1, _calc_1);
		row4_2 = _mm_xor_si128(row4_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_3, 13);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 13));
		row4_3 = _mm_xor_si128(row4_3, _shift_left);
		row4_2 = _mm_xor_si128(row4_2, _calc_2);
		row4_3 = _mm_xor_si128(row4_3, _calc_3);

		/* fourth row */
		_calc_1 = _mm_add_epi32(row4_1, row3_1);
		_calc_2 = _mm_add_epi32(row4_2, row3_2);
		_shift_left = _mm_slli_epi32(_calc_1, 18);
		_calc_3 = _mm_add_epi32(row4_3, row3_3);
		row1_1 = _mm_xor_si128(row1_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_2, 18);
		row1_1 = _mm_xor_si128(row1_1, _calc_1);
		row1_2 = _mm_xor_si128(row1_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_3, 18);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 18));
		row1_3 = _mm_xor_si128(row1_3, _shift_left);
		row1_2 = _mm_xor_si128(row1_2, _calc_2);
		row1_3 = _mm_xor_si128(row1_3, _calc_3);

		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2_1 = _mm_shuffle_epi32(row2_1,0x93);
		row3_1 = _mm_shuffle_epi32(row3_1,0x4e);
		row4_1 = _mm_shuffle_epi32(row4_1,0x39);
		row2_2 = _mm_shuffle_epi32(row2_2,0x93);
		row3_2 = _mm_shuffle_epi32(row3_2,0x4e);
		row4_2 = _mm_shuffle_epi32(row4_2,0x39);
		row2_3 = _mm_shuffle_epi32(row2_3,0x93);
		row3_3 = _mm_shuffle_epi32(row3_3,0x4e);
		row4_3 = _mm_shuffle_epi32(row4_3,0x39);
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column */
		_calc_1 = _mm_add_epi32(row1_1, row2_1);
		_calc_2 = _mm_add_epi32(row1_2, row2_2);
		_shift_left = _mm_slli_epi32(_calc_1, 7);
		_calc_3 = _mm_add_epi32(row1_3, row2_3);
		row4_1 = _mm_xor_si128(row4_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_2, 7);
		row4_1 = _mm_xor_si128(row4_1, _calc_1);
		row4_2 = _mm_xor_si128(row4_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_3, 7);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 7));
		row4_3 = _mm_xor_si128(row4_3, _shift_left);
		row4_2 = _mm_xor_si128(row4_2, _calc_2);
		row4_3 = _mm_xor_si128(row4_3, _calc_3);

		/* second column */
		_calc_1 = _mm_add_epi32(row4_1, row1_1);
		_calc_2 = _mm_add_epi32(row4_2, row1_2);
		_shift_left = _mm_slli_epi32(_calc_1, 9);
		_calc_3 = _mm_add_epi32(row4_3, row1_3);
		row3_1 = _mm_xor_si128(row3_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_2, 9);
		row3_1 = _mm_xor_si128(row3_1, _calc_1);
		row3_2 = _mm_xor_si128(row3_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_3, 9);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 9));
		row3_3 = _mm_xor_si128(row3_3, _shift_left);
		row3_2 = _mm_xor_si128(row3_2, _calc_2);
		row3_3 = _mm_xor_si128(row3_3, _calc_3);

		/* third column */
		_calc_1 = _mm_add_epi32(row3_1, row4_1);
		_calc_2 = _mm_add_epi32(row3_2, row4_2);
		_shift_left = _mm_slli_epi32(_calc_1, 13);
		_calc_3 = _mm_add_epi32(row3_3, row4_3);
		row2_1 = _mm_xor_si128(row2_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_2, 13);
		row2_1 = _mm_xor_si128(row2_1, _calc_1);
		row2_2 = _mm_xor_si128(row2_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_3, 13);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 13));
		row2_3 = _mm_xor_si128(row2_3, _shift_left);
		row2_2 = _mm_xor_si128(row2_2, _calc_2);
		row2_3 = _mm_xor_si128(row2_3, _calc_3);

		/* fourth column */
		_calc_1 = _mm_add_epi32(row2_1, row3_1);
		_calc_2 = _mm_add_epi32(row2_2, row3_2);
		_shift_left = _mm_slli_epi32(_calc_1, 18);
		_calc_3 = _mm_add_epi32(row2_3, row3_3);
		row1_1 = _mm_xor_si128(row1_1, _shift_left);
		_calc_1 = _mm_srli_epi32(_calc_1,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_2, 18);
		row1_1 = _mm_xor_si128(row1_1, _calc_1);
		row1_2 = _mm_xor_si128(row1_2, _shift_left);
		_calc_2 = _mm_srli_epi32(_calc_2,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_3, 18);
		_calc_3 = _mm_srli_epi32(_calc_3,(32 - 18));
		row1_3 = _mm_xor_si128(row1_3, _shift_left);
		row1_2 = _mm_xor_si128(row1_2, _calc_2);
		row1_3 = _mm_xor_si128(row1_3, _calc_3);

		// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2_1 = _mm_shuffle_epi32(row2_1,0x39);
		row3_1 = _mm_shuffle_epi32(row3_1,0x4e);
		row4_1 = _mm_shuffle_epi32(row4_1,0x93);
		row2_2 = _mm_shuffle_epi32(row2_2,0x39);
		row3_2 = _mm_shuffle_epi32(row3_2,0x4e);
		row4_2 = _mm_shuffle_epi32(row4_2,0x93);
		row2_3 = _mm_shuffle_epi32(row2_3,0x39);
		row3_3 = _mm_shuffle_epi32(row3_3,0x4e);
		row4_3 = _mm_shuffle_epi32(row4_3,0x93);
	// end transpose
	}
	row1_1 = _mm_add_epi32(*calc_18_1,row1_1);
	row2_1 = _mm_add_epi32(*calc_7_1, row2_1);
	row3_1 = _mm_add_epi32(*calc_9_1, row3_1);
	row4_1 = _mm_add_epi32(*calc_13_1, row4_1);

	row1_2 = _mm_add_epi32(*calc_18_2,row1_2);
	row2_2 = _mm_add_epi32(*calc_7_2, row2_2);
	row3_2 = _mm_add_epi32(*calc_9_2, row3_2);
	row4_2 = _mm_add_epi32(*calc_13_2, row4_2);

	row1_3 = _mm_add_epi32(*calc_18_3,row1_3);
	row2_3 = _mm_add_epi32(*calc_7_3, row2_3);
	row3_3 = _mm_add_epi32(*calc_9_3, row3_3);
	row4_3 = _mm_add_epi32(*calc_13_3, row4_3);

	*calc_18_1 = *calc_1_1;
	*calc_7_1 = *calc_2_1;
	*calc_9_1 = *calc_3_1;
	*calc_13_1 = *calc_4_1;
	*calc_18_2 = *calc_1_2;
	*calc_7_2 = *calc_2_2;
	*calc_9_2 = *calc_3_2;
	*calc_13_2 = *calc_4_2;
	*calc_18_3 = *calc_1_3;
	*calc_7_3 = *calc_2_3;
	*calc_9_3 = *calc_3_3;
	*calc_13_3 = *calc_4_3;

	*calc_1_1 = row1_1;
	*calc_2_1 = row2_1;
	*calc_3_1 = row3_1;
	*calc_4_1 = row4_1;
	*calc_1_2 = row1_2;
	*calc_2_2 = row2_2;
	*calc_3_2 = row3_2;
	*calc_4_2 = row4_2;
	*calc_1_3 = row1_3;
	*calc_2_3 = row2_3;
	*calc_3_3 = row3_3;
	*calc_4_3 = row4_3;
}

static inline void scrypt_core_r2_sidm_X3(__m128i *X_1, __m128i *X_2, __m128i *X_3, uint32_t Loops, uint32_t double_rounds)
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


	__m128i _calc5;
	__m128i _calc6;
	__m128i _calc7;
	__m128i _calc8;

	/* transpose the data from *X_1 */
	_calc5 =_mm_blend_epi16(*calc_31_1, *calc_33_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32_1, *calc_34_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33_1, *calc_31_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34_1, *calc_32_1, 0x0f);
	*calc_31_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21_1, *calc_23_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22_1, *calc_24_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23_1, *calc_21_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24_1, *calc_22_1, 0x0f);
	*calc_21_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11_1, *calc_13_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12_1, *calc_14_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13_1, *calc_11_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14_1, *calc_12_1, 0x0f);
	*calc_11_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1_1, *calc_3_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2_1, *calc_4_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3_1, *calc_1_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4_1, *calc_2_1, 0x0f);
	*calc_1_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	/* transpose the data from *X_2 */
	_calc5 =_mm_blend_epi16(*calc_31_2, *calc_33_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32_2, *calc_34_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33_2, *calc_31_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34_2, *calc_32_2, 0x0f);
	*calc_31_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21_2, *calc_23_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22_2, *calc_24_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23_2, *calc_21_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24_2, *calc_22_2, 0x0f);
	*calc_21_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11_2, *calc_13_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12_2, *calc_14_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13_2, *calc_11_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14_2, *calc_12_2, 0x0f);
	*calc_11_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1_2, *calc_3_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2_2, *calc_4_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3_2, *calc_1_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4_2, *calc_2_2, 0x0f);
	*calc_1_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	/* transpose the data from *X_3 */
	_calc5 =_mm_blend_epi16(*calc_31_3, *calc_33_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32_3, *calc_34_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33_3, *calc_31_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34_3, *calc_32_3, 0x0f);
	*calc_31_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21_3, *calc_23_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22_3, *calc_24_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23_3, *calc_21_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24_3, *calc_22_3, 0x0f);
	*calc_21_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11_3, *calc_13_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12_3, *calc_14_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13_3, *calc_11_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14_3, *calc_12_3, 0x0f);
	*calc_11_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1_3, *calc_3_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2_3, *calc_4_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3_3, *calc_1_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4_3, *calc_2_3, 0x0f);
	*calc_1_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);


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

		xor_salsa_sidm_X3( calc_1_1, calc_2_1, calc_3_1, calc_4_1, calc_31_1,calc_32_1,calc_33_1,calc_34_1,
						   calc_1_2, calc_2_2, calc_3_2, calc_4_2, calc_31_2,calc_32_2,calc_33_2,calc_34_2,
				           calc_1_3, calc_2_3, calc_3_3, calc_4_3, calc_31_3,calc_32_3,calc_33_3,calc_34_3,
				           double_rounds);

		scratch_1[i * 16 + 4] = *calc_11_1;		scratch_1[i * 16 + 5] = *calc_12_1;
		scratch_1[i * 16 + 6] = *calc_13_1;		scratch_1[i * 16 + 7] = *calc_14_1;
		scratch_2[i * 16 + 4] = *calc_11_2;		scratch_2[i * 16 + 5] = *calc_12_2;
		scratch_2[i * 16 + 6] = *calc_13_2;		scratch_2[i * 16 + 7] = *calc_14_2;
		scratch_3[i * 16 + 4] = *calc_11_3;		scratch_3[i * 16 + 5] = *calc_12_3;
		scratch_3[i * 16 + 6] = *calc_13_3;		scratch_3[i * 16 + 7] = *calc_14_3;

		xor_salsa_sidm_X3( calc_11_1, calc_12_1, calc_13_1, calc_14_1, calc_1_1,calc_2_1,calc_3_1,calc_4_1,
						   calc_11_2, calc_12_2, calc_13_2, calc_14_2, calc_1_2,calc_2_2,calc_3_2,calc_4_2,
						   calc_11_3, calc_12_3, calc_13_3, calc_14_3, calc_1_3,calc_2_3,calc_3_3,calc_4_3,
						   double_rounds);

		scratch_1[i * 16 +  8] = *calc_21_1;	scratch_1[i * 16 +  9] = *calc_22_1;
		scratch_1[i * 16 + 10] = *calc_23_1;	scratch_1[i * 16 + 11] = *calc_24_1;
		scratch_2[i * 16 +  8] = *calc_21_2;	scratch_2[i * 16 +  9] = *calc_22_2;
		scratch_2[i * 16 + 10] = *calc_23_2;	scratch_2[i * 16 + 11] = *calc_24_2;
		scratch_3[i * 16 +  8] = *calc_21_3;	scratch_3[i * 16 +  9] = *calc_22_3;
		scratch_3[i * 16 + 10] = *calc_23_3;	scratch_3[i * 16 + 11] = *calc_24_3;

		xor_salsa_sidm_swap_X3(  calc_21_1, calc_22_1, calc_23_1, calc_24_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
							     calc_21_2, calc_22_2, calc_23_2, calc_24_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							     calc_21_3, calc_22_3, calc_23_3, calc_24_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							     double_rounds);

		xor_salsa_sidm_X3(  calc_31_1, calc_32_1, calc_33_1, calc_34_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
							calc_31_2, calc_32_2, calc_33_2, calc_34_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							calc_31_3, calc_32_3, calc_33_3, calc_34_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							double_rounds);
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


		xor_salsa_sidm_X3( calc_1_1, calc_2_1, calc_3_1, calc_4_1, calc_31_1,calc_32_1,calc_33_1,calc_34_1,
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

		xor_salsa_sidm_X3( calc_11_1, calc_12_1, calc_13_1, calc_14_1, calc_1_1,calc_2_1,calc_3_1,calc_4_1,
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

		xor_salsa_sidm_swap_X3(  calc_21_1, calc_22_1, calc_23_1, calc_24_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
							     calc_21_2, calc_22_2, calc_23_2, calc_24_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							     calc_21_3, calc_22_3, calc_23_3, calc_24_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							     double_rounds);

		xor_salsa_sidm_X3(  calc_31_1, calc_32_1, calc_33_1, calc_34_1, calc_11_1,calc_12_1,calc_13_1,calc_14_1,
							calc_31_2, calc_32_2, calc_33_2, calc_34_2, calc_11_2,calc_12_2,calc_13_2,calc_14_2,
							calc_31_3, calc_32_3, calc_33_3, calc_34_3, calc_11_3,calc_12_3,calc_13_3,calc_14_3,
							double_rounds);
	}
// 1
	_calc5 =_mm_blend_epi16(*calc_31_1, *calc_33_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32_1, *calc_34_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33_1, *calc_31_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34_1, *calc_32_1, 0x0f);
	*calc_31_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21_1, *calc_23_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22_1, *calc_24_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23_1, *calc_21_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24_1, *calc_22_1, 0x0f);
	*calc_21_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11_1, *calc_13_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12_1, *calc_14_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13_1, *calc_11_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14_1, *calc_12_1, 0x0f);
	*calc_11_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1_1, *calc_3_1, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2_1, *calc_4_1, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3_1, *calc_1_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4_1, *calc_2_1, 0x0f);
	*calc_1_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2_1 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3_1 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4_1 = _mm_blend_epi16(_calc8, _calc7, 0xcc);
// 2
	_calc5 =_mm_blend_epi16(*calc_31_2, *calc_33_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32_2, *calc_34_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33_2, *calc_31_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34_2, *calc_32_2, 0x0f);
	*calc_31_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21_2, *calc_23_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22_2, *calc_24_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23_2, *calc_21_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24_2, *calc_22_2, 0x0f);
	*calc_21_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11_2, *calc_13_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12_2, *calc_14_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13_2, *calc_11_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14_2, *calc_12_2, 0x0f);
	*calc_11_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1_2, *calc_3_2, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2_2, *calc_4_2, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3_2, *calc_1_2, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4_2, *calc_2_2, 0x0f);
	*calc_1_2 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3_2 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4_2 = _mm_blend_epi16(_calc8, _calc7, 0xcc);
// 3
	_calc5 =_mm_blend_epi16(*calc_31_3, *calc_33_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_32_3, *calc_34_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_33_3, *calc_31_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_34_3, *calc_32_3, 0x0f);
	*calc_31_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_32_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_33_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_34_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_21_3, *calc_23_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_22_3, *calc_24_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_23_3, *calc_21_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_24_3, *calc_22_3, 0x0f);
	*calc_21_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_22_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_23_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_24_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_11_3, *calc_13_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_12_3, *calc_14_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_13_3, *calc_11_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_14_3, *calc_12_3, 0x0f);
	*calc_11_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_12_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_13_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_14_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1_3, *calc_3_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2_3, *calc_4_3, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3_3, *calc_1_3, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4_3, *calc_2_3, 0x0f);
	*calc_1_3 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2_3 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4_3 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

}

//---------------------------------------------------------------------------------------------
// end threefold
//---------------------------------------------------------------------------------------------

static inline void scrypt_core_sidm(__m128i *X , uint32_t Loops, uint32_t double_rounds)
{
	uint32_t i, j;
	__m128i scratch[Loops * 8];

	__m128i *calc_1 = (__m128i*) &X[0];
	__m128i *calc_2 = (__m128i*) &X[1];
	__m128i *calc_3 = (__m128i*) &X[2];
	__m128i *calc_4 = (__m128i*) &X[3];

	__m128i *calc_11 = (__m128i*) &X[4];
	__m128i *calc_21 = (__m128i*) &X[5];
	__m128i *calc_31 = (__m128i*) &X[6];
	__m128i *calc_41 = (__m128i*) &X[7];

	__m128i _calc5;
	__m128i _calc6;
	__m128i _calc7;
	__m128i _calc8;

	/* transpose the data from *X */
	_calc5 =_mm_blend_epi16(*calc_11, *calc_31, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_21, *calc_41, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_31, *calc_11, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_41, *calc_21, 0x0f);
	*calc_11 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_21 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_31 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_41 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1, *calc_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2, *calc_4, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3, *calc_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4, *calc_2, 0x0f);
	*calc_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	for (i = 0; i < Loops; i++) {
		scratch[i * 8 + 0] = *calc_1;
		scratch[i * 8 + 1] = *calc_2;
		scratch[i * 8 + 2] = *calc_3;
		scratch[i * 8 + 3] = *calc_4;
		scratch[i * 8 + 4] = *calc_11;
		scratch[i * 8 + 5] = *calc_21;
		scratch[i * 8 + 6] = *calc_31;
		scratch[i * 8 + 7] = *calc_41;

		xor_salsa_sidm( calc_1, calc_2, calc_3, calc_4,calc_11,calc_21,calc_31,calc_41, double_rounds);
		xor_salsa_sidm(calc_11,calc_21,calc_31,calc_41, calc_1, calc_2, calc_3, calc_4, double_rounds);
	}

	for (i = 0; i < Loops; i++) {
		j = 8 * (_mm_extract_epi16(*calc_11,0x00) & (Loops-1));

		*calc_1 = _mm_xor_si128(*calc_1, scratch[j]);
		*calc_2 = _mm_xor_si128(*calc_2, scratch[j+1]);
		*calc_3 = _mm_xor_si128(*calc_3, scratch[j+2]);
		*calc_4 = _mm_xor_si128(*calc_4, scratch[j+3]);
		*calc_11 = _mm_xor_si128(*calc_11, scratch[j+4]);
		*calc_21 = _mm_xor_si128(*calc_21, scratch[j+5]);
		*calc_31 = _mm_xor_si128(*calc_31, scratch[j+6]);
		*calc_41 = _mm_xor_si128(*calc_41, scratch[j+7]);

		xor_salsa_sidm( calc_1, calc_2, calc_3, calc_4,calc_11,calc_21,calc_31,calc_41, double_rounds);
		xor_salsa_sidm(calc_11,calc_21,calc_31,calc_41, calc_1, calc_2, calc_3, calc_4, double_rounds);
	}

	_calc5 =_mm_blend_epi16(*calc_11, *calc_31, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_21, *calc_41, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_31, *calc_11, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_41, *calc_21, 0x0f);
	*calc_11 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_21 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_31 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_41 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1, *calc_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2, *calc_4, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3, *calc_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4, *calc_2, 0x0f);
	*calc_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4 = _mm_blend_epi16(_calc8, _calc7, 0xcc);
}

static inline void xor_salsa_sidm_3way(__m128i *calc_11, __m128i *calc_21, __m128i *calc_31,
									uint32_t double_rounds)
{
	int i;
	__m128i _calc_x1;
	__m128i _calc_x2;
	__m128i _calc_x3;
	__m128i _shift_left;
	__m128i X1[4];
	__m128i X2[4];
	__m128i X3[4];

	X1[0] = calc_11[0];
	X1[1] = calc_11[1];
	X1[2] = calc_11[2];
	X1[3] = calc_11[3];

	X2[0] = calc_21[0];
	X2[1] = calc_21[1];
	X2[2] = calc_21[2];
	X2[3] = calc_21[3];

	X3[0] = calc_31[0];
	X3[1] = calc_31[1];
	X3[2] = calc_31[2];
	X3[3] = calc_31[3];

	for (i = 0; i < double_rounds; i++) {

		/* first row  X[3]=f(X0,X1) */
 		_calc_x1 = _mm_add_epi32(X1[0], X1[1]);     //X[0] and X[1]
 		_calc_x2 = _mm_add_epi32(X2[0], X2[1]);     //X[0] and X[1]
		_shift_left = _mm_slli_epi32(_calc_x1, 7);
 		_calc_x3 = _mm_add_epi32(X3[0], X3[1]);     //X[0] and X[1]
		X1[3] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_x2, 7);
		X1[3] ^= _calc_x1;
		X2[3] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_x3, 7);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 7));
		X3[3] ^= _shift_left;
		X2[3] ^= _calc_x2;
		X3[3] ^= _calc_x3;

		/* second rows X[2]=f(X3,X0) */
 		_calc_x1 = _mm_add_epi32(X1[3], X1[0]);     //X[3] and X[0]
 		_calc_x2 = _mm_add_epi32(X2[3], X2[0]);     //X[3] and X[0]
		_shift_left = _mm_slli_epi32(_calc_x1, 9);
 		_calc_x3 = _mm_add_epi32(X3[3], X3[0]);     //X[3] and X[0]
		X1[2] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_x2, 9);
		X1[2] ^= _calc_x1;
		X2[2] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_x3, 9);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 9));
		X3[2] ^= _shift_left;
		X2[2] ^= _calc_x2;
		X3[2] ^= _calc_x3;

		/* third rows X[1]=f(X2,X3) */
 		_calc_x1 = _mm_add_epi32(X1[2], X1[3]);     //X[2] and X[3]
 		_calc_x2 = _mm_add_epi32(X2[2], X2[3]);     //X[2] and X[3]
		_shift_left = _mm_slli_epi32(_calc_x1, 13);
 		_calc_x3 = _mm_add_epi32(X3[2], X3[3]);     //X[2] and X[3]
		X1[1] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_x2, 13);
		X1[1] ^= _calc_x1;
		X2[1] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_x3, 13);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 13));
		X3[1] ^= _shift_left;
		X2[1] ^= _calc_x2;
		X3[1] ^= _calc_x3;

		/* fourth rows X[0]=f(X1,X2) */
 		_calc_x1 = _mm_add_epi32(X1[1], X1[2]);     //X[1] and X[2]
 		_calc_x2 = _mm_add_epi32(X2[1], X2[2]);     //X[1] and X[2]
		_shift_left = _mm_slli_epi32(_calc_x1, 18);
 		_calc_x3 = _mm_add_epi32(X3[1], X3[2]);     //X[1] and X[2]
		X1[0] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_x2, 18);
		X1[0] ^= _calc_x1;
		X2[0] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_x3, 18);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 18));
		X3[0] ^= _shift_left;
		X2[0] ^= _calc_x2;
		X3[0] ^= _calc_x3;

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		X1[3] = _mm_shuffle_epi32(X1[3],0x93);    //x[3]
		X2[3] = _mm_shuffle_epi32(X2[3],0x93);    //x[3]
		X3[3] = _mm_shuffle_epi32(X3[3],0x93);    //x[3]
		X1[2] = _mm_shuffle_epi32(X1[2],0x4e);    //x[2]
		X2[2] = _mm_shuffle_epi32(X2[2],0x4e);    //x[2]
		X3[2] = _mm_shuffle_epi32(X3[2],0x4e);    //x[2]
		X1[1] = _mm_shuffle_epi32(X1[1],0x39);    //x[1]
		X2[1] = _mm_shuffle_epi32(X2[1],0x39);    //x[1]
		X3[1] = _mm_shuffle_epi32(X3[1],0x39);    //x[1]
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column X[1]=f(X0,X3) */
 		_calc_x1 = _mm_add_epi32(X1[0], X1[3]);     //X[0] and X[3]
 		_calc_x2 = _mm_add_epi32(X2[0], X2[3]);     //X[0] and X[3]
		_shift_left = _mm_slli_epi32(_calc_x1, 7);
 		_calc_x3 = _mm_add_epi32(X3[0], X3[3]);     //X[0] and X[3]
		X1[1] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_x2, 7);
		X1[1] ^= _calc_x1;
		X2[1] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 7));
		_shift_left = _mm_slli_epi32(_calc_x3, 7);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 7));
		X3[1] ^= _shift_left;
		X2[1] ^= _calc_x2;
		X3[1] ^= _calc_x3;

		/* second column X[2]=f(X1,X0) */
 		_calc_x1 = _mm_add_epi32(X1[1], X1[0]);     //X[1] and X[0]
 		_calc_x2 = _mm_add_epi32(X2[1], X2[0]);     //X[1] and X[0]
		_shift_left = _mm_slli_epi32(_calc_x1, 9);
 		_calc_x3 = _mm_add_epi32(X3[1], X3[0]);     //X[1] and X[0]
		X1[2] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_x2, 9);
		X1[2] ^= _calc_x1;
		X2[2] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 9));
		_shift_left = _mm_slli_epi32(_calc_x3, 9);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 9));
		X3[2] ^= _shift_left;
		X2[2] ^= _calc_x2;
		X3[2] ^= _calc_x3;

		/* third column  X[3]=f(X2,X1) */
 		_calc_x1 = _mm_add_epi32(X1[2], X1[1]);     //X[2] and X[1]
 		_calc_x2 = _mm_add_epi32(X2[2], X2[1]);     //X[2] and X[1]
		_shift_left = _mm_slli_epi32(_calc_x1, 13);
 		_calc_x3 = _mm_add_epi32(X3[2], X3[1]);     //X[2] and X[1]
		X1[3] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_x2, 13);
		X1[3] ^= _calc_x1;
		X2[3] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 13));
		_shift_left = _mm_slli_epi32(_calc_x3, 13);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 13));
		X3[3] ^= _shift_left;
		X2[3] ^= _calc_x2;
		X3[3] ^= _calc_x3;

		/* fourth column  X[0]=f(X3,X2) */
 		_calc_x1 = _mm_add_epi32(X1[3], X1[2]);     //X[3] and X[2]
 		_calc_x2 = _mm_add_epi32(X2[3], X2[2]);     //X[3] and X[2]
		_shift_left = _mm_slli_epi32(_calc_x1, 18);
 		_calc_x3 = _mm_add_epi32(X3[3], X3[2]);     //X[3] and X[2]
		X1[0] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_x2, 18);
		X1[0] ^= _calc_x1;		//X[0]
		X2[0] ^= _shift_left;
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 18));
		_shift_left = _mm_slli_epi32(_calc_x3, 18);
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 18));
		X3[0] ^= _shift_left;
		X2[0] ^= _calc_x2;		//X[0]
		X3[0] ^= _calc_x3;		//X[0]

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		X1[3] = _mm_shuffle_epi32(X1[3],0x39);    //x[3]
		X2[3] = _mm_shuffle_epi32(X2[3],0x39);    //x[3]
		X3[3] = _mm_shuffle_epi32(X3[3],0x39);    //x[3]
		X1[2] = _mm_shuffle_epi32(X1[2],0x4e);    //x[2]
		X2[2] = _mm_shuffle_epi32(X2[2],0x4e);    //x[2]
		X3[2] = _mm_shuffle_epi32(X3[2],0x4e);    //x[2]
		X1[1] = _mm_shuffle_epi32(X1[1],0x93);    //x[1]
		X2[1] = _mm_shuffle_epi32(X2[1],0x93);    //x[1]
		X3[1] = _mm_shuffle_epi32(X3[1],0x93);    //x[1]

	// end transpose
	}

	calc_11[0] = _mm_add_epi32(calc_11[0], X1[0]);
	calc_11[1] = _mm_add_epi32(calc_11[1], X1[1]);
	calc_11[2] = _mm_add_epi32(calc_11[2], X1[2]);
	calc_11[3] = _mm_add_epi32(calc_11[3], X1[3]);

	calc_21[0] = _mm_add_epi32(calc_21[0], X2[0]);
	calc_21[1] = _mm_add_epi32(calc_21[1], X2[1]);
	calc_21[2] = _mm_add_epi32(calc_21[2], X2[2]);
	calc_21[3] = _mm_add_epi32(calc_21[3], X2[3]);

	calc_31[0] = _mm_add_epi32(calc_31[0], X3[0]);
	calc_31[1] = _mm_add_epi32(calc_31[1], X3[1]);
	calc_31[2] = _mm_add_epi32(calc_31[2], X3[2]);
	calc_31[3] = _mm_add_epi32(calc_31[3], X3[3]);

}


static inline void scrypt_core_sidm_3way(__m128i *X , uint32_t Loops, uint32_t double_rounds)
{
	uint32_t i, j;

	__m128i scratch[Loops * 8 * 3];
	__m128i *SourcePtr = (__m128i*) X;
	__m128i X11[4];
	__m128i X12[4];
	__m128i X21[4];
	__m128i X22[4];
	__m128i X31[4];
	__m128i X32[4];

	__m128i *calc_11 = (__m128i*) X11;
	__m128i *calc_21 = (__m128i*) X21;
	__m128i *calc_31 = (__m128i*) X31;
	__m128i *calc_12 = (__m128i*) X12;
	__m128i *calc_22 = (__m128i*) X22;
	__m128i *calc_32 = (__m128i*) X32;

	__m128i _calc5;
	__m128i _calc6;
	__m128i _calc7;
	__m128i _calc8;

	// working with multiple pointers for the scratch-pad results in minimized instruction count.
    __m128i *scratchPrt1 = &scratch[0];
    __m128i *scratchPrt2 = &scratch[1];
    __m128i *scratchPrt3 = &scratch[2];
    __m128i *scratchPrt4 = &scratch[3];
    __m128i *scratchPrt5 = &scratch[4];
    __m128i *scratchPrt6 = &scratch[5];
    __m128i *scratchPrt7 = &scratch[6];
    __m128i *scratchPrt8 = &scratch[7];

	/* transpose the data from *X1x */
	_calc5 =_mm_blend_epi16(SourcePtr[0], SourcePtr[2], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[1], SourcePtr[3], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[2], SourcePtr[0], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[3], SourcePtr[1], 0x0f);
	calc_11[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	calc_11[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	calc_11[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	calc_11[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(SourcePtr[4], SourcePtr[6], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[5], SourcePtr[7], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[6], SourcePtr[4], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[7], SourcePtr[5], 0x0f);
	calc_12[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	calc_12[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	calc_12[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	calc_12[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	/* transpose the data from *X2x */
	_calc5 =_mm_blend_epi16(SourcePtr[8], SourcePtr[10], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[9], SourcePtr[11], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[10], SourcePtr[8], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[11], SourcePtr[9], 0x0f);
	calc_21[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	calc_21[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	calc_21[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	calc_21[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(SourcePtr[12], SourcePtr[14], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[13], SourcePtr[15], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[14], SourcePtr[12], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[15], SourcePtr[13], 0x0f);
	calc_22[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	calc_22[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	calc_22[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	calc_22[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	/* transpose the data from *X3x */
	_calc5 =_mm_blend_epi16(SourcePtr[16], SourcePtr[18], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[17], SourcePtr[19], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[18], SourcePtr[16], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[19], SourcePtr[17], 0x0f);
	calc_31[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	calc_31[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	calc_31[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	calc_31[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(SourcePtr[20], SourcePtr[22], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[21], SourcePtr[23], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[22], SourcePtr[20], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[23], SourcePtr[21], 0x0f);
	calc_32[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	calc_32[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	calc_32[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	calc_32[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	for (i = 0; i < Loops; i++) {
		for (j=0; j<4; j++){
			scratch[i * 24 +  0 + j] = calc_11[j];
			scratch[i * 24 +  4 + j] = calc_12[j];
			scratch[i * 24 +  8 + j] = calc_21[j];
			scratch[i * 24 + 12 + j] = calc_22[j];
			scratch[i * 24 + 16 + j] = calc_31[j];
			scratch[i * 24 + 20 + j] = calc_32[j];
		}
		calc_11[0] ^= calc_12[0];
		calc_11[1] ^= calc_12[1];
		calc_11[2] ^= calc_12[2];
		calc_11[3] ^= calc_12[3];

		calc_21[0] ^= calc_22[0];
		calc_21[1] ^= calc_22[1];
		calc_21[2] ^= calc_22[2];
		calc_21[3] ^= calc_22[3];

		calc_31[0] ^= calc_32[0];
		calc_31[1] ^= calc_32[1];
		calc_31[2] ^= calc_32[2];
		calc_31[3] ^= calc_32[3];

		xor_salsa_sidm_3way(calc_11, calc_21, calc_31, double_rounds);

		calc_12[0] ^= calc_11[0];
		calc_12[1] ^= calc_11[1];
		calc_12[2] ^= calc_11[2];
		calc_12[3] ^= calc_11[3];

		calc_22[0] ^= calc_21[0];
		calc_22[1] ^= calc_21[1];
		calc_22[2] ^= calc_21[2];
		calc_22[3] ^= calc_21[3];

		calc_32[0] ^= calc_31[0];
		calc_32[1] ^= calc_31[1];
		calc_32[2] ^= calc_31[2];
		calc_32[3] ^= calc_31[3];

		xor_salsa_sidm_3way(calc_12, calc_22, calc_32, double_rounds);
	}
	for (i = 0; i < Loops; i++) {
		j = 24 * (_mm_extract_epi16(calc_12[0],0x00) & (Loops-1));

		calc_11[0] ^=  scratchPrt1[j];
		calc_11[1] ^=  scratchPrt2[j];
		calc_11[2] ^=  scratchPrt3[j];
		calc_11[3] ^=  scratchPrt4[j];
		calc_12[0] ^=  scratchPrt5[j];
		calc_12[1] ^=  scratchPrt6[j];
		calc_12[2] ^=  scratchPrt7[j];
		calc_12[3] ^=  scratchPrt8[j];

		j = 8 + 24 * (_mm_extract_epi16(calc_22[0],0x00) & (Loops-1));

		calc_21[0] ^=  scratchPrt1[j];
		calc_21[1] ^=  scratchPrt2[j];
		calc_21[2] ^=  scratchPrt3[j];
		calc_21[3] ^=  scratchPrt4[j];
		calc_22[0] ^=  scratchPrt5[j];
		calc_22[1] ^=  scratchPrt6[j];
		calc_22[2] ^=  scratchPrt7[j];
		calc_22[3] ^=  scratchPrt8[j];

		j = 16 + 24 * (_mm_extract_epi16(calc_32[0],0x00) & (Loops-1));

		calc_31[0] ^=  scratchPrt1[j];
		calc_31[1] ^=  scratchPrt2[j];
		calc_31[2] ^=  scratchPrt3[j];
		calc_31[3] ^=  scratchPrt4[j];
		calc_32[0] ^=  scratchPrt5[j];
		calc_32[1] ^=  scratchPrt6[j];
		calc_32[2] ^=  scratchPrt7[j];
		calc_32[3] ^=  scratchPrt8[j];

		calc_11[0] ^= calc_12[0];
		calc_11[1] ^= calc_12[1];
		calc_11[2] ^= calc_12[2];
		calc_11[3] ^= calc_12[3];

		calc_21[0] ^= calc_22[0];
		calc_21[1] ^= calc_22[1];
		calc_21[2] ^= calc_22[2];
		calc_21[3] ^= calc_22[3];

		calc_31[0] ^= calc_32[0];
		calc_31[1] ^= calc_32[1];
		calc_31[2] ^= calc_32[2];
		calc_31[3] ^= calc_32[3];

		xor_salsa_sidm_3way(calc_11, calc_21, calc_31, double_rounds);

		calc_12[0] ^= calc_11[0];
		calc_12[1] ^= calc_11[1];
		calc_12[2] ^= calc_11[2];
		calc_12[3] ^= calc_11[3];

		calc_22[0] ^= calc_21[0];
		calc_22[1] ^= calc_21[1];
		calc_22[2] ^= calc_21[2];
		calc_22[3] ^= calc_21[3];

		calc_32[0] ^= calc_31[0];
		calc_32[1] ^= calc_31[1];
		calc_32[2] ^= calc_31[2];
		calc_32[3] ^= calc_31[3];

		xor_salsa_sidm_3way(calc_12, calc_22, calc_32, double_rounds);
	}
// return the valueś to X
	_calc5 =_mm_blend_epi16(calc_11[0], calc_11[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_11[1], calc_11[3], 0x0f);
	_calc7 =_mm_blend_epi16(calc_11[2], calc_11[0], 0xf0);
	_calc8 =_mm_blend_epi16(calc_11[3], calc_11[1], 0x0f);
	SourcePtr[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_12[0], calc_12[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_12[1], calc_12[3], 0x0f);
	_calc7 =_mm_blend_epi16(calc_12[2], calc_12[0], 0xf0);
	_calc8 =_mm_blend_epi16(calc_12[3], calc_12[1], 0x0f);
	SourcePtr[4] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[5] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[6] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[7] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_21[0], calc_21[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_21[1], calc_21[3], 0x0f);
	_calc7 =_mm_blend_epi16(calc_21[2], calc_21[0], 0xf0);
	_calc8 =_mm_blend_epi16(calc_21[3], calc_21[1], 0x0f);
	SourcePtr[8] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[9] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[10] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[11] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_22[0], calc_22[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_22[1], calc_22[3], 0x0f);
	_calc7 =_mm_blend_epi16(calc_22[2], calc_22[0], 0xf0);
	_calc8 =_mm_blend_epi16(calc_22[3], calc_22[1], 0x0f);
	SourcePtr[12] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[13] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[14] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[15] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_31[0], calc_31[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_31[1], calc_31[3], 0x0f);
	_calc7 =_mm_blend_epi16(calc_31[2], calc_31[0], 0xf0);
	_calc8 =_mm_blend_epi16(calc_31[3], calc_31[1], 0x0f);
	SourcePtr[16] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[17] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[18] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[19] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_32[0], calc_32[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_32[1], calc_32[3], 0x0f);
	_calc7 =_mm_blend_epi16(calc_32[2], calc_32[0], 0xf0);
	_calc8 =_mm_blend_epi16(calc_32[3], calc_32[1], 0x0f);
	SourcePtr[20] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[21] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[22] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[23] = _mm_blend_epi16(_calc8, _calc7, 0xcc);
}

#endif

