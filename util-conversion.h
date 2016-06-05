/*
 * util-conversion.h
 *
 *  Created on: Mar 3, 2016
 *      Author: yakov
 */

#ifndef UTIL_CONVERSION_H_
#define UTIL_CONVERSION_H_
#include <stdbool.h>
#include <stddef.h>

bool b58dec(unsigned char *bin, size_t binsz, const char *b58);
int b58check(unsigned char *bin, size_t binsz, const char *b58);


#endif /* UTIL_CONVERSION_H_ */
