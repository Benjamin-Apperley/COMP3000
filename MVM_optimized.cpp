#include "MVM.h"

int main()
{
	
	struct timespec start, end;
	uint64_t diff;
	double gflops;
	int outcome;

	initialization_MVM();
	
	clock_gettime(CLOCK_MONOTONIC, &start);

	for(int i = 0; i < TIMES; i++)
	{
		MVM_AVX();
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	gflops = (double) ARITHMETICAL_OPS / (diff / TIMES);
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);
	printf("%f GigaFLOPS achieved\n", gflops);
	
	outcome = Compare_MVM();	

	if (outcome == 0)
		printf("\n\n\r -----  output is correct -----\n\r");
	else
		printf("\n\n\r ----- output is INcorrect -----\n\r");

	return 0;

}

void initialization_MVM() {

	float e = 0.1234, p = 0.7264, r = 0.11;

	for (unsigned int i = 0; i != M; i++)
		for (unsigned int j = 0; j != M; j++)
			A1[i][j] = ((i - j) % 9) + p;

	for (unsigned int j = 0; j != M; j++) {
		Y[j] = 0.0;
		test1[j] = 0.0;
		X[j] = (j % 7) + r;
	}
}


unsigned short int MVM_AVX() {

	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, num0, num1, num2, num3, num4, num5;
	__m128 xmm1, xmm2;
	float temp;
	int i, j;

	for (i = 0; i < M; i++) {
		num1 = _mm256_setzero_ps();

		for (j = 0; j < M;/*((M / 8) * 8);*/ j += 8) {

			num5 = _mm256_load_ps(X + j);
			num0 = _mm256_load_ps(&A1[i][j]);
			num1 = _mm256_fmadd_ps(num0, num5, num1);
		}

		ymm2 = _mm256_permute2f128_ps(num1, num1, 1);
		num1 = _mm256_add_ps(num1, ymm2);
		num1 = _mm256_hadd_ps(num1, num1);
		num1 = _mm256_hadd_ps(num1, num1);
		xmm2 = _mm256_extractf128_ps(num1, 0);
		_mm_store_ss(Y + i, xmm2);

		for (; j < M; j++) { 
			Y[i] += A1[i][j] * X[j];
		}

	}

	return 1;
}


unsigned short int Compare_MVM() {

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			test1[i] += A1[i][j] * X[j];
		}
	}

	for (int j = 0; j < M; j++)
		if (equal(Y[j], test1[j]) == 1) {
			//printf("\n j=%d\n", j);
			return 1;
		}

	return 0;
}


unsigned short int equal(float const a, float const b) {
	float temp = a - b;
	//printf("\n %f  %f", a, b);
	if (fabs(temp) < EPSILON)
		return 0; //success
	else
		return 1;
}
