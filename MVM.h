#pragma once

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <stdint.h>	/* for uint64 definition */
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>
#include <time.h>

//MVM initialization 
#define M 64
static float  X[M], Y[M], test1[M], A1[M][M] __attribute__((aligned(64))); 

void initialization_MVM();

unsigned short int MVM_default();
unsigned short int Compare_MVM();
unsigned short int equal(float const a, float const b);

#define TIMES 1000
#define BILLION 1000000000L
#define ARITHMETICAL_OPS M*16
#define EPSILON 0.01
