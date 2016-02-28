/*
 * cpu-miner-work.h
 *
 *  Created on: Feb 28, 2016
 *      Author: yakov
 */

#ifndef CPU_MINER_WORK_H_
#define CPU_MINER_WORK_H_
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <jansson.h>
#include <curl/curl.h>

struct work {
	uint32_t data[32];
	uint32_t target[8];

	int height;
	char *txs;
	char *workid;

	char *job_id;
	size_t xnonce2_len;
	unsigned char *xnonce2;
};

void work_free(struct work *w);
void work_copy(struct work *dest, const struct work *src);
bool work_decode(const json_t *val, struct work *work);
bool gbt_work_decode(const json_t *val, struct work *work);
bool submit_upstream_work(CURL *curl, struct work *work);

#endif /* CPU_MINER_WORK_H_ */
