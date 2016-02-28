/*
 * cpu-miner-opt.h
 *
 *  Created on: Feb 28, 2016
 *      Author: yakov
 */

#ifndef CPU_MINER_OPT_H_
#define CPU_MINER_OPT_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum algos {
	ALGO_NEOSCRYPT = 0,		/* NeoScrypt(128, 2, 1) with Salsa20/20 and ChaCha20/20 */
	ALGO_ALTSCRYPT = 1,		/* Scrypt(1024, 1, 1) with Salsa20/8 through NeoScrypt */
	ALGO_SCRYPT = 2,		/* Scrypt(1024, 1, 1) with Salsa20/8 */
	ALGO_SHA256D = 3,		/* SHA-256d */
	ALGO_SIZE = 4
};

extern const char *algo_names[];

extern bool opt_debug;
extern bool opt_protocol;
extern bool opt_benchmark;
extern bool opt_redirect;
extern bool opt_background;
extern bool opt_quiet;
extern int opt_retries;
extern int opt_fail_pause;
extern int opt_timeout;
extern int opt_scantime;
extern const bool opt_time;
extern enum algos opt_algo;
extern uint8_t opt_neoscrypt_profile;
extern uint8_t opt_neoscrypt_asm;
extern uint8_t opt_nfactor;
extern int opt_n_threads;
extern char *opt_cert;
extern char *opt_proxy;
extern long opt_proxy_type;





#endif /* CPU_MINER_OPT_H_ */
