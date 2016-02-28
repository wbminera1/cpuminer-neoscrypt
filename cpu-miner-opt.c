/*
 * cpu-miner-opt.c
 *
 *  Created on: Feb 28, 2016
 *      Author: yakov
 */
#include "cpu-miner-opt.h"

const char *algo_names[] = {
	[ALGO_NEOSCRYPT]	= "neoscrypt",
	[ALGO_ALTSCRYPT]	= "altscrypt",
	[ALGO_SCRYPT]		= "scrypt",
	[ALGO_SHA256D]		= "sha256d",
};

bool opt_debug = false;
bool opt_protocol = false;
bool opt_benchmark = false;
bool opt_redirect = true;
bool opt_background = false;
bool opt_quiet = false;
int opt_retries = -1;
int opt_fail_pause = 30;
int opt_timeout = 0;
int opt_scantime = 5;
const bool opt_time = true;
enum algos opt_algo = ALGO_NEOSCRYPT;
uint8_t opt_neoscrypt_profile = 0;
uint8_t opt_neoscrypt_asm = 0;
uint8_t opt_nfactor = 6;
int opt_n_threads;
char *opt_cert;
char *opt_proxy;
long opt_proxy_type;
