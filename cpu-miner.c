/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2014 pooler
 * Copyright 2014-2015 John Doering <ghostlander@phoenixcoin.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "cpuminer-config.h"
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#ifdef WIN32
#include <windows.h>
typedef unsigned long ulong;
typedef unsigned int uint;
#else
#include <errno.h>
#include <signal.h>
#include <sys/resource.h>
#if HAVE_SYS_SYSCTL_H
#include <sys/types.h>
#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <sys/sysctl.h>
#endif
#endif
#include <jansson.h>
#include <curl/curl.h>
#include "compat.h"
#include "miner.h"
#include "version.h"

#include "cpu-miner-opt.h"
#include "cpu-miner-work.h"
#include "neoscrypt.h"

#define PROGRAM_NAME		"minerd"
#define LP_SCANTIME		60

#ifdef __linux /* Linux specific policy and affinity management */
#include <sched.h>
static inline void drop_policy(void)
{
	struct sched_param param;
	param.sched_priority = 0;

#ifdef SCHED_IDLE
	if (unlikely(sched_setscheduler(0, SCHED_IDLE, &param) == -1))
#endif
#ifdef SCHED_BATCH
		sched_setscheduler(0, SCHED_BATCH, &param);
#endif
}

static inline void affine_to_cpu(int id, int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(set), &set);
}
#elif defined(__FreeBSD__) /* FreeBSD specific policy and affinity management */
#include <sys/cpuset.h>
static inline void drop_policy(void)
{
}

static inline void affine_to_cpu(int id, int cpu)
{
	cpuset_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(cpuset_t), &set);
}
#else
static inline void drop_policy(void)
{
}

static inline void affine_to_cpu(int id, int cpu)
{
}
#endif
		
enum workio_commands {
	WC_GET_WORK,
	WC_SUBMIT_WORK,
};

struct workio_cmd {
	enum workio_commands	cmd;
	struct thr_info		*thr;
	union {
		struct work	*work;
	} u;
};


bool want_longpoll = true;
bool have_longpoll = false;
bool have_gbt = true;
bool allow_getwork = true;
bool want_stratum = true;
bool have_stratum = false;
bool use_syslog = false;
int num_processors;
char *rpc_url;
char *rpc_userpass;
char *rpc_user, *rpc_pass;
int pk_script_size;
unsigned char pk_script[25];
char coinbase_sig[101] = "";
struct thr_info *thr_info;
int work_thr_id;
int longpoll_thr_id = -1;
int stratum_thr_id = -1;
struct work_restart *work_restart = NULL;
struct stratum_ctx stratum;

pthread_mutex_t applog_lock;
pthread_mutex_t stats_lock;

static unsigned long accepted_count = 0L;
static unsigned long rejected_count = 0L;
static double *thr_hashrates;

#ifdef HAVE_GETOPT_LONG
#include <getopt.h>
#else
struct option {
	const char *name;
	int has_arg;
	int *flag;
	int val;
};
#endif

static char const usage[] = "\
Usage: " PROGRAM_NAME " [OPTIONS]\n\
Options:\n\
  -a, --algo=ALGO       specify the algorithm to use\n\
      neoscrypt  NeoScrypt(128, 2, 1) with Salsa20/20 and ChaCha20/20 (default)\n\
      altscrypt  Scrypt(1024, 1, 1) with Salsa20/8 through NeoScrypt\n\
      scrypt     Scrypt(1024, 1, 1) with Salsa20/8\n\
      sha256d    SHA-256d\n"
#if (ASM)
"\
  -e, --engine=N        choose a NeoScrypt hashing engine\n\
      0          integer (default)\n\
      1          SSE2\n"
#if (MINER_4WAY)
"\
      2          SSE2 4-way\n"
#endif
#endif
"\
  -o, --url=URL         URL of mining server\n\
  -O, --userpass=U:P    username:password pair for mining server\n\
  -u, --user=USERNAME   username for mining server\n\
  -p, --pass=PASSWORD   password for mining server\n\
      --cert=FILE       certificate for mining server using SSL\n\
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy\n\
  -t, --threads=N       number of miner threads (default: number of processors)\n\
  -r, --retries=N       number of times to retry if a network call fails\n\
                          (default: retry indefinitely)\n\
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 30)\n\
  -T, --timeout=N       timeout for long polling, in seconds (default: none)\n\
  -s, --scantime=N      upper bound on time spent scanning current work when\n\
                          long polling is unavailable, in seconds (default: 5)\n\
      --coinbase-addr=ADDR  payout address for solo mining\n\
      --coinbase-sig=TEXT  data to insert in the coinbase when possible\n\
      --no-longpoll     disable long polling support\n\
      --no-getwork      disable getwork support\n\
      --no-gbt          disable getblocktemplate support\n\
      --no-stratum      disable X-Stratum support\n\
      --no-redirect     ignore requests to change the URL of the mining server\n\
  -q, --quiet           disable per-thread hashmeter output\n\
  -D, --debug           enable debug output\n\
  -P, --protocol-dump   verbose dump of protocol-level activities\n"
#ifdef HAVE_SYSLOG_H
"\
  -S, --syslog          use system log for output messages\n"
#endif
#ifndef WIN32
"\
  -B, --background      run the miner in the background\n"
#endif
"\
      --benchmark       run in offline benchmark mode\n\
  -c, --config=FILE     load a JSON-format configuration file\n\
  -V, --version         display version information and exit\n\
  -h, --help            display this help text and exit\n\
";

static char const short_options[] =
#ifndef WIN32
	"B"
#endif
#ifdef HAVE_SYSLOG_H
	"S"
#endif
	"a:e:n:c:Dhp:Px:qr:R:s:t:T:o:u:O:V";

static struct option const options[] = {
	{ "algo", 1, NULL, 'a' },
#if (ASM)
	{ "engine", 1, NULL, 'e' },
#endif
#ifndef WIN32
	{ "background", 0, NULL, 'B' },
#endif
	{ "benchmark", 0, NULL, 1005 },
	{ "cert", 1, NULL, 1001 },
	{ "coinbase-addr", 1, NULL, 1013 },
	{ "coinbase-sig", 1, NULL, 1015 },
	{ "config", 1, NULL, 'c' },
	{ "debug", 0, NULL, 'D' },
	{ "help", 0, NULL, 'h' },
	{ "no-gbt", 0, NULL, 1011 },
	{ "no-getwork", 0, NULL, 1010 },
	{ "no-longpoll", 0, NULL, 1003 },
	{ "no-redirect", 0, NULL, 1009 },
	{ "no-stratum", 0, NULL, 1007 },
	{ "pass", 1, NULL, 'p' },
	{ "protocol-dump", 0, NULL, 'P' },
	{ "proxy", 1, NULL, 'x' },
	{ "quiet", 0, NULL, 'q' },
	{ "retries", 1, NULL, 'r' },
	{ "retry-pause", 1, NULL, 'R' },
	{ "scantime", 1, NULL, 's' },
#ifdef HAVE_SYSLOG_H
	{ "syslog", 0, NULL, 'S' },
#endif
	{ "threads", 1, NULL, 't' },
	{ "timeout", 1, NULL, 'T' },
	{ "url", 1, NULL, 'o' },
	{ "user", 1, NULL, 'u' },
	{ "userpass", 1, NULL, 'O' },
	{ "version", 0, NULL, 'V' },
	{ 0, 0, 0, 0 }
};


struct work g_work;
time_t g_work_time;
pthread_mutex_t g_work_lock;
bool submit_old = false;
char *lp_id;

bool jobj_binary(const json_t *obj, const char *key,
			void *buf, size_t buflen)
{
	const char *hexstr;
	json_t *tmp;

	tmp = json_object_get(obj, key);
	if (unlikely(!tmp)) {
		applog(LOG_ERR, "JSON key '%s' not found", key);
		return false;
	}
	hexstr = json_string_value(tmp);
	if (unlikely(!hexstr)) {
		applog(LOG_ERR, "JSON key '%s' is not a string", key);
		return false;
	}
	if (!hex2bin(buf, hexstr, buflen))
		return false;

	return true;
}

void share_result(int result, const char *reason)
{
	char s[345];
	double hashrate;
	int i;

	hashrate = 0.;
	pthread_mutex_lock(&stats_lock);
	for (i = 0; i < opt_n_threads; i++)
		hashrate += thr_hashrates[i];
	result ? accepted_count++ : rejected_count++;
	pthread_mutex_unlock(&stats_lock);
	
	sprintf(s, hashrate >= 1e6 ? "%.0f" : "%.3f", 1e-3 * hashrate);
	applog(LOG_INFO, "accepted: %lu/%lu (%.3f%%), %s KH/s %s",
		   accepted_count,
		   accepted_count + rejected_count,
		   100. * accepted_count / (accepted_count + rejected_count),
		   s,
		   result ? "(yay!!!)" : "(booooo)");

	if (opt_debug && reason)
		applog(LOG_DEBUG, "DEBUG: reject reason: %s", reason);
}

static const char *getwork_req =
	"{\"method\": \"getwork\", \"params\": [], \"id\":0}\r\n";

#define GBT_CAPABILITIES "[\"coinbasetxn\", \"coinbasevalue\", \"longpoll\", \"workid\"]"

static const char *gbt_req =
	"{\"method\": \"getblocktemplate\", \"params\": [{\"capabilities\": "
	GBT_CAPABILITIES "}], \"id\":0}\r\n";
static const char *gbt_lp_req =
	"{\"method\": \"getblocktemplate\", \"params\": [{\"capabilities\": "
	GBT_CAPABILITIES ", \"longpollid\": \"%s\"}], \"id\":0}\r\n";

static bool get_upstream_work(CURL *curl, struct work *work)
{
	json_t *val;
	int err;
	bool rc;
	struct timeval tv_start, tv_end, diff;

start:
	gettimeofday(&tv_start, NULL);
	val = json_rpc_call(curl, rpc_url, rpc_userpass,
			    have_gbt ? gbt_req : getwork_req,
			    &err, have_gbt ? JSON_RPC_QUIET_404 : 0);
	gettimeofday(&tv_end, NULL);

	if (have_stratum) {
		if (val)
			json_decref(val);
		return true;
	}

	if (!have_gbt && !allow_getwork) {
		applog(LOG_ERR, "No usable protocol");
		if (val)
			json_decref(val);
		return false;
	}

	if (have_gbt && allow_getwork && !val && err == CURLE_OK) {
		applog(LOG_INFO, "getblocktemplate failed, falling back to getwork");
		have_gbt = false;
		goto start;
	}

	if (!val)
		return false;

	if (have_gbt) {
		rc = gbt_work_decode(json_object_get(val, "result"), work);
		if (!have_gbt) {
			json_decref(val);
			goto start;
		}
	} else
		rc = work_decode(json_object_get(val, "result"), work);

	if (opt_debug && rc) {
		timeval_subtract(&diff, &tv_end, &tv_start);
		applog(LOG_DEBUG, "DEBUG: got new work in %d ms",
		       diff.tv_sec * 1000 + diff.tv_usec / 1000);
	}

	json_decref(val);

	return rc;
}

static void workio_cmd_free(struct workio_cmd *wc)
{
	if (!wc)
		return;

	switch (wc->cmd) {
	case WC_SUBMIT_WORK:
		work_free(wc->u.work);
		free(wc->u.work);
		break;
	default: /* do nothing */
		break;
	}

	memset(wc, 0, sizeof(*wc));	/* poison */
	free(wc);
}

static bool workio_get_work(struct workio_cmd *wc, CURL *curl)
{
	struct work *ret_work;
	int failures = 0;

	ret_work = calloc(1, sizeof(*ret_work));
	if (!ret_work)
		return false;

	/* obtain new work from bitcoin via JSON-RPC */
	while (!get_upstream_work(curl, ret_work)) {
		if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
			applog(LOG_ERR, "json_rpc_call failed, terminating workio thread");
			free(ret_work);
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "json_rpc_call failed, retry after %d seconds",
			opt_fail_pause);
		sleep(opt_fail_pause);
	}

	/* send work to requesting thread */
	if (!tq_push(wc->thr->q, ret_work))
		free(ret_work);

	return true;
}

static bool workio_submit_work(struct workio_cmd *wc, CURL *curl)
{
	int failures = 0;

	/* submit solution to bitcoin via JSON-RPC */
	while (!submit_upstream_work(curl, wc->u.work)) {
		if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
			applog(LOG_ERR, "...terminating workio thread");
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "...retry after %d seconds",
			opt_fail_pause);
		sleep(opt_fail_pause);
	}

	return true;
}

static void *workio_thread(void *userdata)
{
	struct thr_info *mythr = userdata;
	CURL *curl;
	bool ok = true;

	curl = curl_easy_init();
	if (unlikely(!curl)) {
		applog(LOG_ERR, "CURL initialization failed");
		return NULL;
	}

	while (ok) {
		struct workio_cmd *wc;

		/* wait for workio_cmd sent to us, on our queue */
		wc = tq_pop(mythr->q, NULL);
		if (!wc) {
			ok = false;
			break;
		}

		/* process workio_cmd */
		switch (wc->cmd) {
		case WC_GET_WORK:
			ok = workio_get_work(wc, curl);
			break;
		case WC_SUBMIT_WORK:
			ok = workio_submit_work(wc, curl);
			break;

		default:		/* should never happen */
			ok = false;
			break;
		}

		workio_cmd_free(wc);
	}

	tq_freeze(mythr->q);
	curl_easy_cleanup(curl);

	return NULL;
}

static bool get_work(struct thr_info *thr, struct work *work)
{
	struct workio_cmd *wc;
	struct work *work_heap;

	if (opt_benchmark) {
		memset(work->data, 0x55, 76);
		work->data[17] = swab32(time(NULL));
		memset(work->data + 19, 0x00, 52);
		work->data[20] = 0x80000000;
		work->data[31] = 0x00000280;
		memset(work->target, 0x00, sizeof(work->target));
		return true;
	}

	/* fill out work request message */
	wc = calloc(1, sizeof(*wc));
	if (!wc)
		return false;

	wc->cmd = WC_GET_WORK;
	wc->thr = thr;

	/* send work request to workio thread */
	if (!tq_push(thr_info[work_thr_id].q, wc)) {
		workio_cmd_free(wc);
		return false;
	}

	/* wait for response, a unit of work */
	work_heap = tq_pop(thr->q, NULL);
	if (!work_heap)
		return false;

	/* copy returned work into storage provided by caller */
	memcpy(work, work_heap, sizeof(*work));
	free(work_heap);

	return true;
}

static bool submit_work(struct thr_info *thr, const struct work *work_in)
{
	struct workio_cmd *wc;
	
	/* fill out work request message */
	wc = calloc(1, sizeof(*wc));
	if (!wc)
		return false;

	wc->u.work = malloc(sizeof(*work_in));
	if (!wc->u.work)
		goto err_out;

	wc->cmd = WC_SUBMIT_WORK;
	wc->thr = thr;
	work_copy(wc->u.work, work_in);

	/* send solution to workio thread */
	if (!tq_push(thr_info[work_thr_id].q, wc))
		goto err_out;

	return true;

err_out:
	workio_cmd_free(wc);
	return false;
}

static void stratum_gen_work(struct stratum_ctx *sctx, struct work *work)
{
	unsigned char merkle_root[64];
	int i;

	pthread_mutex_lock(&sctx->work_lock);

	free(work->job_id);
	work->job_id = strdup(sctx->job.job_id);
	work->xnonce2_len = sctx->xnonce2_size;
	work->xnonce2 = realloc(work->xnonce2, sctx->xnonce2_size);
	memcpy(work->xnonce2, sctx->job.xnonce2, sctx->xnonce2_size);

	/* Generate merkle root */
	sha256d(merkle_root, sctx->job.coinbase, sctx->job.coinbase_size);
	for (i = 0; i < sctx->job.merkle_count; i++) {
		memcpy(merkle_root + 32, sctx->job.merkle[i], 32);
		sha256d(merkle_root, merkle_root, 64);
	}
	
	/* Increment extranonce2 */
	for (i = 0; i < sctx->xnonce2_size && !++sctx->job.xnonce2[i]; i++);

    /* Assemble block header;
     * reverse byte order for NeoScrypt */
    memset(work->data, 0, 128);
    if(opt_algo != ALGO_NEOSCRYPT) {
        work->data[0] = le32dec(sctx->job.version);
        for(i = 0; i < 8; i++)
          work->data[1 + i] = le32dec((uint32_t *) sctx->job.prevhash + i);
        for(i = 0; i < 8; i++)
          work->data[9 + i] = be32dec((uint32_t *) merkle_root + i);
        work->data[17] = le32dec(sctx->job.ntime);
        work->data[18] = le32dec(sctx->job.nbits);
    } else {
        work->data[0] = be32dec(sctx->job.version);
        for(i = 0; i < 8; i++)
          work->data[1 + i] = be32dec((uint32_t *) sctx->job.prevhash + i);
        for(i = 0; i < 8; i++)
          work->data[9 + i] = le32dec((uint32_t *) merkle_root + i);
        work->data[17] = be32dec(sctx->job.ntime);
        work->data[18] = be32dec(sctx->job.nbits);
    }
    work->data[20] = 0x80000000;
    work->data[31] = 0x00000280;

	pthread_mutex_unlock(&sctx->work_lock);

	if (opt_debug) {
		char *xnonce2str = abin2hex(work->xnonce2, work->xnonce2_len);
		applog(LOG_DEBUG, "DEBUG: job_id='%s' extranonce2=%s ntime=%08x",
		       work->job_id, xnonce2str, swab32(work->data[17]));
		free(xnonce2str);
	}

        if(opt_algo == ALGO_SHA256D)
          diff_to_target(work->target, sctx->job.diff);
        else
          diff_to_target(work->target, sctx->job.diff / 65536.0);
}

bool fulltest_le(const uint *hash, const uint *target) {
    uint i;
    bool rc;

    for(i = 7; i >= 0; i--) {
        if(hash[i] > target[i]) {
            rc = false;
            break;
        }
        if(hash[i] < target[i]) {
            rc = true;
            break;
        }
    }

    if(opt_debug) {
    	uint8_t hash_str[65], target_str[65];

        bin2hex(hash_str, (uint8_t *) hash, 32);
        bin2hex(target_str, (uint8_t *) target, 32);

        applog(LOG_DEBUG, "DEBUG (little endian): %s\nHash:   %sx0\nTarget: %sx0",
          rc ? "hash <= target"
             : "hash > target (false positive)",
               hash_str, target_str);

    }

    return(rc);
}

static int scanhash_neoscrypt(int thr_id, uint *pdata, const uint *ptarget,
  uint max_nonce, ulong *hashes_done, uint profile) {
    uint hash[8];
    const uint targint = ptarget[7];
    uint start_nonce = pdata[19], inc_nonce = 1;

     while((pdata[19] < max_nonce) && !work_restart[thr_id].restart) {

        neoscrypt((uint8_t *) pdata, (uint8_t *) hash, profile);

        /* Quick hash check */
        if(hash[7] <= targint) {
            /* Complete hash check */
            if(fulltest_le(hash, ptarget)) {
                *hashes_done = pdata[19] - start_nonce;
                return(1);
            }
        }

        pdata[19] += inc_nonce;

    } 

    *hashes_done = pdata[19] - inc_nonce - start_nonce;
    return(0);
}

static int scanhash_altscrypt(int thr_id, uint *pdata, const uint *ptarget,
  uint max_nonce, ulong *hashes_done, uint profile) {
    uint hash[8], data[20];
    const uint targint = ptarget[7];
    uint start_nonce = pdata[19], inc_nonce = 1;
    uint i;

    /* Convert BE to LE */
    for(i = 0; i < 19; i++)
      data[i] = be32dec(&pdata[i]);

    data[19] = pdata[19];

     while((data[19] < max_nonce) && !work_restart[thr_id].restart) {

        neoscrypt((uint8_t *) data, (uint8_t *) hash, profile);

        if(hash[7] <= targint) {
            if(fulltest_le(hash, ptarget)) {
                /* LE to BE and write back */
                be32enc(&pdata[19], data[19]);
                *hashes_done = data[19] - start_nonce;
                return(1);
            }
        }

        data[19] += inc_nonce;

    }

    *hashes_done = data[19] - inc_nonce - start_nonce;
    return(0);
}

#if (ASM) && (MINER_4WAY)

static int scanhash_neoscrypt_4way(int thr_id, uint *pdata,
  const uint *ptarget, uint max_nonce, ulong *hashes_done) {
    uint hash[32];
    const uint targint = ptarget[7];
    uint start_nonce = pdata[19];
    const uint inc_nonce = 4;

     while((pdata[19] <= (max_nonce - inc_nonce))
       && !work_restart[thr_id].restart) {

        neoscrypt_4way((uint8_t *) pdata, (uint8_t *) hash, 0);

        if(hash[7] <= targint) {
            if(fulltest_le(&hash[0], ptarget)) {
                *hashes_done = pdata[19] - start_nonce;
                return(1);
            }
        }

        if(hash[15] <= targint) {
            if(fulltest_le(&hash[8], ptarget)) {
                pdata[19] += 1;
                *hashes_done = pdata[19] - start_nonce;
                return(1);
            }
        }

        if(hash[23] <= targint) {
            if(fulltest_le(&hash[16], ptarget)) {
                pdata[19] += 2;
                *hashes_done = pdata[19] - start_nonce;
                return(1);
            }
        }

        if(hash[31] <= targint) {
            if(fulltest_le(&hash[24], ptarget)) {
                pdata[19] += 3;
                *hashes_done = pdata[19] - start_nonce;
                return(1);
            }
        }

        pdata[19] += inc_nonce;

    }

    *hashes_done = pdata[19] - inc_nonce - start_nonce;
    return(0);
}

static int scanhash_altscrypt_4way(int thr_id, uint *pdata,
  const uint *ptarget, uint max_nonce, ulong *hashes_done) {
    uint hash[32], data[20];
    const uint targint = ptarget[7];
    uint start_nonce = pdata[19];
    const uint inc_nonce = 4;
    uint i;

    for(i = 0; i < 19; i++)
      data[i] = be32dec(&pdata[i]);

    data[19] = pdata[19];

     while((data[19] <= (max_nonce - inc_nonce))
       && !work_restart[thr_id].restart) {

        neoscrypt_4way((uint8_t *) data, (uint8_t *) hash, 1);

        if(hash[7] <= targint) {
            if(fulltest_le(&hash[0], ptarget)) {
                be32enc(&pdata[19], data[19]);
                *hashes_done = data[19] - start_nonce;
                return(1);
            }
        }

        if(hash[15] <= targint) {
            if(fulltest_le(&hash[8], ptarget)) {
                data[19] += 1;
                be32enc(&pdata[19], data[19]);
                *hashes_done = data[19] - start_nonce;
                return(1);
            }
        }

        if(hash[23] <= targint) {
            if(fulltest_le(&hash[16], ptarget)) {
                data[19] += 2;
                be32enc(&pdata[19], data[19]);
                *hashes_done = data[19] - start_nonce;
                return(1);
            }
        }

        if(hash[31] <= targint) {
            if(fulltest_le(&hash[24], ptarget)) {
                data[19] += 3;
                be32enc(&pdata[19], data[19]);
                *hashes_done = data[19] - start_nonce;
                return(1);
            }
        }

        data[19] += inc_nonce;

    }

    *hashes_done = data[19] - inc_nonce - start_nonce;
    return(0);
}


#endif

static void *miner_thread(void *userdata)
{
	struct thr_info *mythr = userdata;
	int thr_id = mythr->id;
	struct work work = {{0}};
	uint32_t max_nonce;
	uint32_t end_nonce = 0xffffffffU / opt_n_threads * (thr_id + 1) - 0x20;
	unsigned char *scratchbuf = NULL;
	char s[16];
	int i;

	/* Set worker threads to nice 19 and then preferentially to SCHED_IDLE
	 * and if that fails, then SCHED_BATCH. No need for this to be an
	 * error if it fails */
	if (!opt_benchmark) {
		setpriority(PRIO_PROCESS, 0, 19);
		drop_policy();
	}

	/* Cpu affinity only makes sense if the number of threads is a multiple
	 * of the number of CPUs */
	if (num_processors > 1 && opt_n_threads % num_processors == 0) {
		if (!opt_quiet)
			applog(LOG_INFO, "Binding thread %d to cpu %d",
			       thr_id, thr_id % num_processors);
		affine_to_cpu(thr_id, thr_id % num_processors);
	}
	
	if (opt_algo == ALGO_SCRYPT)
	{
		scratchbuf = scrypt_buffer_alloc();
	}

	while (1) {
		unsigned long hashes_done;
		struct timeval tv_start, tv_end, diff;
		int64_t max64;
		int rc;

        if(have_stratum) {

            while(time(NULL) >= g_work_time + 120)
              sleep(1);

            while(!stratum.job.diff) {
                applog(LOG_DEBUG, "Waiting for Stratum to set the job difficulty");
                sleep(1);
            }

            pthread_mutex_lock(&g_work_lock);
            if(work.data[19] >= end_nonce && !memcmp(work.data, g_work.data, 76))
              stratum_gen_work(&stratum, &g_work);

        } else {
			int min_scantime = have_longpoll ? LP_SCANTIME : opt_scantime;
			/* obtain new work from internal workio thread */
			pthread_mutex_lock(&g_work_lock);
			if (!have_stratum &&
			    (time(NULL) - g_work_time >= min_scantime ||
			     work.data[19] >= end_nonce)) {
				if (unlikely(!get_work(mythr, &g_work))) {
					applog(LOG_ERR, "work retrieval failed, exiting "
						"mining thread %d", mythr->id);
					pthread_mutex_unlock(&g_work_lock);
					goto out;
				}
				g_work_time = have_stratum ? 0 : time(NULL);
			}
			if (have_stratum) {
				pthread_mutex_unlock(&g_work_lock);
				continue;
			}
		}
		if (memcmp(work.data, g_work.data, 76)) {
			work_free(&work);
			work_copy(&work, &g_work);
			work.data[19] = 0xffffffffU / opt_n_threads * thr_id;
		} else
			work.data[19]++;
		pthread_mutex_unlock(&g_work_lock);
		work_restart[thr_id].restart = 0;
		
		/* adjust max_nonce to meet target scan time */
		if (have_stratum)
			max64 = LP_SCANTIME;
		else
			max64 = g_work_time + (have_longpoll ? LP_SCANTIME : opt_scantime)
			      - time(NULL);
		max64 *= thr_hashrates[thr_id];

                if(max64 <= 0)
                  switch(opt_algo) {

                    case(ALGO_NEOSCRYPT):
                    case(ALGO_ALTSCRYPT):
                    case(ALGO_SCRYPT):
                        max64 = 0x3FFFF;
                        if(opt_nfactor > 3)
                          max64 >>= (opt_nfactor - 3);
                        if(opt_nfactor > 16)
                          max64 = 0xF;
                        break;

                    case(ALGO_SHA256D):
                        max64 = 0x1FFFFF;
                        break;

                }

		if (work.data[19] + max64 > end_nonce)
			max_nonce = end_nonce;
		else
			max_nonce = work.data[19] + max64;
		
		hashes_done = 0;
		gettimeofday(&tv_start, NULL);

        /* Hash and verify against targets */
        switch(opt_algo) {

            case(ALGO_NEOSCRYPT):
#if (ASM) && (MINER_4WAY)
                if(opt_neoscrypt_asm == 2)
                  rc = scanhash_neoscrypt_4way(thr_id, work.data, work.target,
                    max_nonce, &hashes_done);
                else
#endif
                  rc = scanhash_neoscrypt(thr_id, work.data, work.target,
                    max_nonce, &hashes_done, opt_neoscrypt_profile);
                break;

            case(ALGO_ALTSCRYPT):
#if (ASM) && (MINER_4WAY)
                if(opt_neoscrypt_asm == 2)
                  rc = scanhash_altscrypt_4way(thr_id, work.data, work.target,
                    max_nonce, &hashes_done);
                else
#endif
                  rc = scanhash_altscrypt(thr_id, work.data, work.target,
                    max_nonce, &hashes_done, opt_neoscrypt_profile);
                break;

		case ALGO_SCRYPT:
			rc = scanhash_scrypt(thr_id, work.data, scratchbuf, work.target,
			                     max_nonce, &hashes_done);
			break;

		case ALGO_SHA256D:
			rc = scanhash_sha256d(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		default:
			/* should never happen */
			goto out;
		}

		/* record scanhash elapsed time */
		gettimeofday(&tv_end, NULL);
		timeval_subtract(&diff, &tv_end, &tv_start);
		if (diff.tv_usec || diff.tv_sec) {
			pthread_mutex_lock(&stats_lock);
			thr_hashrates[thr_id] =
				hashes_done / (diff.tv_sec + 1e-6 * diff.tv_usec);
			pthread_mutex_unlock(&stats_lock);
		}
		if (!opt_quiet) {
			sprintf(s, thr_hashrates[thr_id] >= 1e6 ? "%.0f" : "%.3f",
				1e-3 * thr_hashrates[thr_id]);
			applog(LOG_INFO, "thread %d: %lu hashes, %s KH/s",
				thr_id, hashes_done, s);
		}
		if (opt_benchmark && thr_id == opt_n_threads - 1) {
			double hashrate = 0.;
			for (i = 0; i < opt_n_threads && thr_hashrates[i]; i++)
				hashrate += thr_hashrates[i];
			if (i == opt_n_threads) {
				sprintf(s, hashrate >= 1e6 ? "%.0f" : "%.3f", 1e-3 * hashrate);
				applog(LOG_INFO, "Total: %s KH/s", s);
			}
		}

		/* if nonce found, submit work */
		if (rc && !opt_benchmark && !submit_work(mythr, &work))
			break;
	}

out:
	tq_freeze(mythr->q);

	return NULL;
}

static void restart_threads(void)
{
	int i;

	for (i = 0; i < opt_n_threads; i++)
		work_restart[i].restart = 1;
}

static void *longpoll_thread(void *userdata)
{
	struct thr_info *mythr = userdata;
	CURL *curl = NULL;
	char *copy_start, *hdr_path = NULL, *lp_url = NULL;
	bool need_slash = false;

	curl = curl_easy_init();
	if (unlikely(!curl)) {
		applog(LOG_ERR, "CURL initialization failed");
		goto out;
	}

start:
	hdr_path = tq_pop(mythr->q, NULL);
	if (!hdr_path)
		goto out;

	/* full URL */
	if (strstr(hdr_path, "://")) {
		lp_url = hdr_path;
		hdr_path = NULL;
	}
	
	/* absolute path, on current server */
	else {
		copy_start = (*hdr_path == '/') ? (hdr_path + 1) : hdr_path;
		if (rpc_url[strlen(rpc_url) - 1] != '/')
			need_slash = true;

		lp_url = malloc(strlen(rpc_url) + strlen(copy_start) + 2);
		if (!lp_url)
			goto out;

		sprintf(lp_url, "%s%s%s", rpc_url, need_slash ? "/" : "", copy_start);
	}

	applog(LOG_INFO, "Long-polling activated for %s", lp_url);

	while (1) {
		json_t *val, *res, *soval;
		char *req = NULL;
		int err;

		if (have_gbt) {
			req = malloc(strlen(gbt_lp_req) + strlen(lp_id) + 1);
			sprintf(req, gbt_lp_req, lp_id);
		}
		val = json_rpc_call(curl, lp_url, rpc_userpass,
				    req ? req : getwork_req, &err,
				    JSON_RPC_LONGPOLL);
		free(req);
		if (have_stratum) {
			if (val)
				json_decref(val);
			goto out;
		}
		if (likely(val)) {
			bool rc;
			applog(LOG_INFO, "LONGPOLL pushed new work");
			res = json_object_get(val, "result");
			soval = json_object_get(res, "submitold");
			submit_old = soval ? json_is_true(soval) : false;
			pthread_mutex_lock(&g_work_lock);
			if (have_gbt)
				rc = gbt_work_decode(res, &g_work);
			else
				rc = work_decode(res, &g_work);
			if (rc) {
				time(&g_work_time);
				restart_threads();
			}
			pthread_mutex_unlock(&g_work_lock);
			json_decref(val);
		} else {
			pthread_mutex_lock(&g_work_lock);
			g_work_time -= LP_SCANTIME;
			pthread_mutex_unlock(&g_work_lock);
			if (err == CURLE_OPERATION_TIMEDOUT) {
				restart_threads();
			} else {
				have_longpoll = false;
				restart_threads();
				free(hdr_path);
				free(lp_url);
				lp_url = NULL;
				sleep(opt_fail_pause);
				goto start;
			}
		}
	}

out:
	free(hdr_path);
	free(lp_url);
	tq_freeze(mythr->q);
	if (curl)
		curl_easy_cleanup(curl);

	return NULL;
}

static bool stratum_handle_response(char *buf)
{
	json_t *val, *err_val, *res_val, *id_val;
	json_error_t err;
	bool ret = false;

	val = JSON_LOADS(buf, &err);
	if (!val) {
		applog(LOG_INFO, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");
	id_val = json_object_get(val, "id");

	if (!id_val || json_is_null(id_val) || !res_val)
		goto out;

	share_result(json_is_true(res_val),
		err_val ? json_string_value(json_array_get(err_val, 1)) : NULL);

	ret = true;
out:
	if (val)
		json_decref(val);

	return ret;
}

static void *stratum_thread(void *userdata)
{
	struct thr_info *mythr = userdata;
	char *s;

	stratum.url = tq_pop(mythr->q, NULL);
	if (!stratum.url)
		goto out;
	applog(LOG_INFO, "Starting Stratum on %s", stratum.url);

	while (1) {
		int failures = 0;

		while (!stratum.curl) {
			pthread_mutex_lock(&g_work_lock);
			g_work_time = 0;
			pthread_mutex_unlock(&g_work_lock);
			restart_threads();

			if (!stratum_connect(&stratum, stratum.url) ||
			    !stratum_subscribe(&stratum) ||
			    !stratum_authorize(&stratum, rpc_user, rpc_pass)) {
				stratum_disconnect(&stratum);
				if (opt_retries >= 0 && ++failures > opt_retries) {
					applog(LOG_ERR, "...terminating workio thread");
					tq_push(thr_info[work_thr_id].q, NULL);
					goto out;
				}
				applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);
				sleep(opt_fail_pause);
			}
		}

		if (stratum.job.job_id &&
		    (!g_work_time || strcmp(stratum.job.job_id, g_work.job_id))) {
			pthread_mutex_lock(&g_work_lock);
			stratum_gen_work(&stratum, &g_work);
			time(&g_work_time);
			pthread_mutex_unlock(&g_work_lock);
			if (stratum.job.clean) {
				applog(LOG_INFO, "Stratum requested work restart");
				restart_threads();
			}
		}
		
		if (!stratum_socket_full(&stratum, 120)) {
			applog(LOG_ERR, "Stratum connection timed out");
			s = NULL;
		} else
			s = stratum_recv_line(&stratum);
		if (!s) {
			stratum_disconnect(&stratum);
			applog(LOG_ERR, "Stratum connection interrupted");
			continue;
		}
		if (!stratum_handle_method(&stratum, s))
			stratum_handle_response(s);
		free(s);
	}

out:
	return NULL;
}

static void show_version_and_exit(void)
{
	printf(PACKAGE_STRING "\n built on " __DATE__ "\n features:"
#if defined(USE_ASM) && defined(__i386__)
		" i386"
#endif
#if defined(USE_ASM) && defined(__x86_64__)
		" x86_64"
#endif
#if defined(USE_ASM) && (defined(__i386__) || defined(__x86_64__))
		" SSE2"
#endif
#if defined(__x86_64__) && defined(USE_AVX)
		" AVX"
#endif
#if defined(__x86_64__) && defined(USE_AVX2)
		" AVX2"
#endif
#if defined(__x86_64__) && defined(USE_XOP)
		" XOP"
#endif
#if defined(USE_ASM) && defined(__arm__) && defined(__APCS_32__)
		" ARM"
#if defined(__ARM_ARCH_5E__) || defined(__ARM_ARCH_5TE__) || \
	defined(__ARM_ARCH_5TEJ__) || defined(__ARM_ARCH_6__) || \
	defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || \
	defined(__ARM_ARCH_6M__) || defined(__ARM_ARCH_6T2__) || \
	defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__) || \
	defined(__ARM_ARCH_7__) || \
	defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || \
	defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__)
		" ARMv5E"
#endif
#if defined(__ARM_NEON__)
		" NEON"
#endif
#endif
		"\n");

	printf("%s\n", curl_version());
#ifdef JANSSON_VERSION
	printf("libjansson %s\n", JANSSON_VERSION);
#endif
	exit(0);
}

static void show_usage_and_exit(int status)
{
	if (status)
		fprintf(stderr, "Try `" PROGRAM_NAME " --help' for more information.\n");
	else
		printf(usage);
	exit(status);
}

static void strhide(char *s)
{
	if (*s) *s++ = 'x';
	while (*s) *s++ = '\0';
}

static void parse_config(json_t *config, char *pname, char *ref);

static void parse_arg(int key, char *arg, char *pname)
{
	char *p;
	int v, i;

	switch(key) {
	case 'a':
		for (i = 0; i < (int)ALGO_SIZE; i++) {
			if (algo_names[i] &&
			    !strcmp(arg, algo_names[i])) {
				opt_algo = i;
				break;
			}
		}
                if((i == ALGO_ALTSCRYPT) || (i == ALGO_SCRYPT))
                  opt_nfactor = 9;
		if (i == (int)ALGO_SIZE) {
			fprintf(stderr, "%s: unknown algorithm -- '%s'\n",
				pname, arg);
			show_usage_and_exit(1);
		}
		break;

        case('e'):
#if (ASM)
            if((opt_algo == ALGO_NEOSCRYPT) || (opt_algo == ALGO_ALTSCRYPT)) {
                v = atoi(arg);
#if (MINER_4WAY)
                if((v < 0) || (v > 2)) {
#else
                if((v < 0) || (v > 1)) {
#endif
                    fprintf(stderr, "%s: incorrect engine %d\n", pname, v);
                    show_usage_and_exit(1);
                }
                opt_neoscrypt_asm = v;
            }
#endif
            break;

        case('n'):
#if !(ASM)
            /* Nfactor is fixed in the NeoScrypt assembly code */
            if((opt_algo == ALGO_NEOSCRYPT) || (opt_algo == ALGO_ALTSCRYPT)) {
                v = atoi(arg);
                /* Nfactor = lb(N) - 1; N = (1 << (Nfactor + 1)) */
                if((v < 0) || (v > 30)) {
                    fprintf(stderr, "%s: incorrect Nfactor %d\n", pname, v);
                    show_usage_and_exit(1);
                }
                opt_nfactor = v;
            }
#endif
            break;

	case 'B':
		opt_background = true;
		break;
	case 'c': {
		json_error_t err;
		json_t *config = JSON_LOAD_FILE(arg, &err);
		if (!json_is_object(config)) {
			if (err.line < 0)
				fprintf(stderr, "%s: %s\n", pname, err.text);
			else
				fprintf(stderr, "%s: %s:%d: %s\n",
					pname, arg, err.line, err.text);
			exit(1);
		}
		parse_config(config, pname, arg);
		json_decref(config);
		break;
	}
	case 'q':
		opt_quiet = true;
		break;
	case 'D':
		opt_debug = true;
		break;
	case 'p':
		free(rpc_pass);
		rpc_pass = strdup(arg);
		strhide(arg);
		break;
	case 'P':
		opt_protocol = true;
		break;
	case 'r':
		v = atoi(arg);
		if (v < -1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_retries = v;
		break;
	case 'R':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_fail_pause = v;
		break;
	case 's':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_scantime = v;
		break;
	case 'T':
		v = atoi(arg);
		if (v < 1 || v > 99999)	/* sanity check */
			show_usage_and_exit(1);
		opt_timeout = v;
		break;
	case 't':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_n_threads = v;
		break;
	case 'u':
		free(rpc_user);
		rpc_user = strdup(arg);
		break;
	case 'o': {			/* --url */
		char *ap, *hp;
		ap = strstr(arg, "://");
		ap = ap ? ap + 3 : arg;
		hp = strrchr(arg, '@');
		if (hp) {
			*hp = '\0';
			p = strchr(ap, ':');
			if (p) {
				free(rpc_userpass);
				rpc_userpass = strdup(ap);
				free(rpc_user);
				rpc_user = calloc(p - ap + 1, 1);
				strncpy(rpc_user, ap, p - ap);
				free(rpc_pass);
				rpc_pass = strdup(++p);
				if (*p) *p++ = 'x';
				v = strlen(hp + 1) + 1;
				memmove(p + 1, hp + 1, v);
				memset(p + v, 0, hp - p);
				hp = p;
			} else {
				free(rpc_user);
				rpc_user = strdup(ap);
			}
			*hp++ = '@';
		} else
			hp = ap;
		if (ap != arg) {
			if (strncasecmp(arg, "http://", 7) &&
			    strncasecmp(arg, "https://", 8) &&
			    strncasecmp(arg, "stratum+tcp://", 14)) {
				fprintf(stderr, "%s: unknown protocol -- '%s'\n",
					pname, arg);
				show_usage_and_exit(1);
			}
			free(rpc_url);
			rpc_url = strdup(arg);
			strcpy(rpc_url + (ap - arg), hp);
		} else {
			if (*hp == '\0' || *hp == '/') {
				fprintf(stderr, "%s: invalid URL -- '%s'\n",
					pname, arg);
				show_usage_and_exit(1);
			}
			free(rpc_url);
			rpc_url = malloc(strlen(hp) + 8);
			sprintf(rpc_url, "http://%s", hp);
		}
		have_stratum = !opt_benchmark && !strncasecmp(rpc_url, "stratum", 7);
		break;
	}
	case 'O':			/* --userpass */
		p = strchr(arg, ':');
		if (!p) {
			fprintf(stderr, "%s: invalid username:password pair -- '%s'\n",
				pname, arg);
			show_usage_and_exit(1);
		}
		free(rpc_userpass);
		rpc_userpass = strdup(arg);
		free(rpc_user);
		rpc_user = calloc(p - arg + 1, 1);
		strncpy(rpc_user, arg, p - arg);
		free(rpc_pass);
		rpc_pass = strdup(++p);
		strhide(p);
		break;
	case 'x':			/* --proxy */
		if (!strncasecmp(arg, "socks4://", 9))
			opt_proxy_type = CURLPROXY_SOCKS4;
		else if (!strncasecmp(arg, "socks5://", 9))
			opt_proxy_type = CURLPROXY_SOCKS5;
#if LIBCURL_VERSION_NUM >= 0x071200
		else if (!strncasecmp(arg, "socks4a://", 10))
			opt_proxy_type = CURLPROXY_SOCKS4A;
		else if (!strncasecmp(arg, "socks5h://", 10))
			opt_proxy_type = CURLPROXY_SOCKS5_HOSTNAME;
#endif
		else
			opt_proxy_type = CURLPROXY_HTTP;
		free(opt_proxy);
		opt_proxy = strdup(arg);
		break;
	case 1001:
		free(opt_cert);
		opt_cert = strdup(arg);
		break;
	case 1005:
		opt_benchmark = true;
		want_longpoll = false;
		want_stratum = false;
		have_stratum = false;
		break;
	case 1003:
		want_longpoll = false;
		break;
	case 1007:
		want_stratum = false;
		break;
	case 1009:
		opt_redirect = false;
		break;
	case 1010:
		allow_getwork = false;
		break;
	case 1011:
		have_gbt = false;
		break;
	case 1013:			/* --coinbase-addr */
		pk_script_size = address_to_script(pk_script, sizeof(pk_script), arg);
		if (!pk_script_size) {
			fprintf(stderr, "%s: invalid address -- '%s'\n",
				pname, arg);
			show_usage_and_exit(1);
		}
		break;
	case 1015:			/* --coinbase-sig */
		if (strlen(arg) + 1 > sizeof(coinbase_sig)) {
			fprintf(stderr, "%s: coinbase signature too long\n", pname);
			show_usage_and_exit(1);
		}
		strcpy(coinbase_sig, arg);
		break;
	case 'S':
		use_syslog = true;
		break;
	case 'V':
		show_version_and_exit();
	case 'h':
		show_usage_and_exit(0);
	default:
		show_usage_and_exit(1);
	}

}

static void parse_config(json_t *config, char *pname, char *ref)
{
	int i;
	char *s;
	json_t *val;

	for (i = 0; i < ARRAY_SIZE(options); i++) {
		if (!options[i].name)
			break;

		val = json_object_get(config, options[i].name);
		if (!val)
			continue;

		if (options[i].has_arg && json_is_string(val)) {
			if (!strcmp(options[i].name, "config")) {
				fprintf(stderr, "%s: %s: option '%s' not allowed here\n",
					pname, ref, options[i].name);
				exit(1);
			}
			s = strdup(json_string_value(val));
			if (!s)
				break;
			parse_arg(options[i].val, s, pname);
			free(s);
		} else if (!options[i].has_arg && json_is_true(val)) {
			parse_arg(options[i].val, "", pname);
		} else {
			fprintf(stderr, "%s: invalid argument for option '%s'\n",
				pname, options[i].name);
			exit(1);
		}
	}
}

static void parse_cmdline(int argc, char *argv[])
{
	int key;

	while (1) {
#if HAVE_GETOPT_LONG
		key = getopt_long(argc, argv, short_options, options, NULL);
#else
		key = getopt(argc, argv, short_options);
#endif
		if (key < 0)
			break;

		parse_arg(key, optarg, argv[0]);
	}
	if (optind < argc) {
		fprintf(stderr, "%s: unsupported non-option argument -- '%s'\n",
			argv[0], argv[optind]);
		show_usage_and_exit(1);
	}
}

#ifndef WIN32
static void signal_handler(int sig)
{
	switch (sig) {
	case SIGHUP:
		applog(LOG_INFO, "SIGHUP received");
		break;
	case SIGINT:
		applog(LOG_INFO, "SIGINT received, exiting");
		exit(0);
		break;
	case SIGTERM:
		applog(LOG_INFO, "SIGTERM received, exiting");
		exit(0);
		break;
	}
}
#endif

int main(int argc, char *argv[])
{
	struct thr_info *thr;
	long flags;
	int i;

    printf("NeoScrypt CPUminer v%u.%u.%u\n",
      VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION);

    /* Processor vector extensions detected */
    uint opt_flags = cpu_vec_exts();

    /* Configure for SSE2-4way by default */
    if(opt_flags & 0x00000020) opt_neoscrypt_asm = 2;

	rpc_user = strdup("");
	rpc_pass = strdup("");

	/* parse command line */
	parse_cmdline(argc, argv);

    if((opt_algo == ALGO_NEOSCRYPT) || (opt_algo == ALGO_ALTSCRYPT)) {
        printf("Engines: ");
#if (ASM)
#if (MINER_4WAY)
        printf("INT SSE2 SSE2-4way (enabled: ");
        if(opt_neoscrypt_asm == 2)
          printf("SSE2-4way)\n");
#else
        printf("INT SSE2 (enabled: ");
#endif
        if(opt_neoscrypt_asm == 1)
          printf("SSE2)\n");
        if(!opt_neoscrypt_asm)
          printf("INT)\n");
#else
        printf("INT (enabled: INT)\n");
#endif

        if(opt_algo == ALGO_NEOSCRYPT) {
            opt_neoscrypt_profile =
              0x80000020 | (opt_nfactor << 8) | ((opt_neoscrypt_asm & 0x1) << 12);
        }

        if(opt_algo == ALGO_ALTSCRYPT) {
            opt_neoscrypt_profile =
              0x80000003 | (opt_nfactor << 8) | ((opt_neoscrypt_asm & 0x1) << 12);
        }

    }

	if (!opt_benchmark && !rpc_url) {
		fprintf(stderr, "%s: no URL supplied\n", argv[0]);
		show_usage_and_exit(1);
	}

	if (!rpc_userpass) {
		rpc_userpass = malloc(strlen(rpc_user) + strlen(rpc_pass) + 2);
		if (!rpc_userpass)
			return 1;
		sprintf(rpc_userpass, "%s:%s", rpc_user, rpc_pass);
	}

	pthread_mutex_init(&applog_lock, NULL);
	pthread_mutex_init(&stats_lock, NULL);
	pthread_mutex_init(&g_work_lock, NULL);
	pthread_mutex_init(&stratum.sock_lock, NULL);
	pthread_mutex_init(&stratum.work_lock, NULL);

	flags = !opt_benchmark && strncmp(rpc_url, "https:", 6)
	      ? (CURL_GLOBAL_ALL & ~CURL_GLOBAL_SSL)
	      : CURL_GLOBAL_ALL;
	if (curl_global_init(flags)) {
		applog(LOG_ERR, "CURL initialization failed");
		return 1;
	}

#ifndef WIN32
	if (opt_background) {
		i = fork();
		if (i < 0) exit(1);
		if (i > 0) exit(0);
		i = setsid();
		if (i < 0)
			applog(LOG_ERR, "setsid() failed (errno = %d)", errno);
		i = chdir("/");
		if (i < 0)
			applog(LOG_ERR, "chdir() failed (errno = %d)", errno);
		signal(SIGHUP, signal_handler);
		signal(SIGINT, signal_handler);
		signal(SIGTERM, signal_handler);
	}
#endif

#if defined(WIN32)
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	num_processors = sysinfo.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_CONF)
	num_processors = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(CTL_HW) && defined(HW_NCPU)
	int req[] = { CTL_HW, HW_NCPU };
	size_t len = sizeof(num_processors);
	sysctl(req, 2, &num_processors, &len, NULL, 0);
#else
	num_processors = 1;
#endif
	if (num_processors < 1)
		num_processors = 1;
	if (!opt_n_threads)
		opt_n_threads = num_processors;

#ifdef HAVE_SYSLOG_H
	if (use_syslog)
		openlog("cpuminer", LOG_PID, LOG_USER);
#endif

	work_restart = calloc(opt_n_threads, sizeof(*work_restart));
	if (!work_restart)
		return 1;

	thr_info = calloc(opt_n_threads + 3, sizeof(*thr));
	if (!thr_info)
		return 1;
	
	thr_hashrates = (double *) calloc(opt_n_threads, sizeof(double));
	if (!thr_hashrates)
		return 1;

	/* init workio thread info */
	work_thr_id = opt_n_threads;
	thr = &thr_info[work_thr_id];
	thr->id = work_thr_id;
	thr->q = tq_new();
	if (!thr->q)
		return 1;

	/* start work I/O thread */
	if (pthread_create(&thr->pth, NULL, workio_thread, thr)) {
		applog(LOG_ERR, "workio thread create failed");
		return 1;
	}

	if (want_longpoll && !have_stratum) {
		/* init longpoll thread info */
		longpoll_thr_id = opt_n_threads + 1;
		thr = &thr_info[longpoll_thr_id];
		thr->id = longpoll_thr_id;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		/* start longpoll thread */
		if (unlikely(pthread_create(&thr->pth, NULL, longpoll_thread, thr))) {
			applog(LOG_ERR, "longpoll thread create failed");
			return 1;
		}
	}
	if (want_stratum) {
		/* init stratum thread info */
		stratum_thr_id = opt_n_threads + 2;
		thr = &thr_info[stratum_thr_id];
		thr->id = stratum_thr_id;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		/* start stratum thread */
		if (unlikely(pthread_create(&thr->pth, NULL, stratum_thread, thr))) {
			applog(LOG_ERR, "stratum thread create failed");
			return 1;
		}

		if (have_stratum)
			tq_push(thr_info[stratum_thr_id].q, strdup(rpc_url));
	}

	/* start mining threads */
	for (i = 0; i < opt_n_threads; i++) {
		thr = &thr_info[i];

		thr->id = i;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		if (unlikely(pthread_create(&thr->pth, NULL, miner_thread, thr))) {
			applog(LOG_ERR, "thread %d create failed", i);
			return 1;
		}
	}

	applog(LOG_INFO, "%d miner threads started, "
		"using '%s' algorithm.",
		opt_n_threads,
		algo_names[opt_algo]);

	/* main loop - simply wait for workio thread to exit */
	pthread_join(thr_info[work_thr_id].pth, NULL);

	applog(LOG_INFO, "workio thread dead, exiting.");

	return 0;
}
