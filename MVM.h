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
#define M 8320
static float  X[M] __attribute__((aligned(64))), Y[M] __attribute__((aligned(64))), test1[M] __attribute__((aligned(64))), A1[M][M] __attribute__((aligned(64))); 

void initialization_MVM();

unsigned short int MVM_default();
unsigned short int MVM_AVX();
unsigned short int MVM_OMP();
unsigned short int MVM_SSE();
unsigned short int MVM_SIMD();
unsigned short int MVM_regBlock_2();
unsigned short int MVM_regBlock_8();
unsigned short int MVM_regBlock_13();
unsigned short int MVM_regBlock_16();
unsigned short int MVM_Looptiling();
unsigned short int MVM_AVX_REG_4();
unsigned short int MVM_AVX_REG_8();
unsigned short int MVM_AVX_REG_13();
unsigned short int MVM_AVX_REG_OMP();
unsigned short int MVM_AVX_REG_OMP_TILE();
unsigned short int MVM_Test();
unsigned short int Compare_MVM();
unsigned short int equal(float const a, float const b);

#define TIMES 1000
#define BILLION 1000000000L
#define ARITHMETICAL_OPS M*M*2
//#define EPSILON 0.01
#define EPSILON 0.5
#define TILEA 64
#define TILEB 4096
