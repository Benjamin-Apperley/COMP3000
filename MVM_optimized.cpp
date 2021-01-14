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
		//MVM_AVX();
		//MVM_OMP();
		//MVM_SSE();
		//MVM_SIMD();
		//MVM_regBlock_2();
		MVM_regBlock_8();
		//MVM_regBlock_13();
		//MVM_regBlock_16();
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

	
	float temp;
	int i, j;
	

	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, num0, num1, num2, num3, num4, num5;
	__m128 xmm1, xmm2;
	
	for (i = 0; i < M; i++) 
	{
		num1 = _mm256_setzero_ps();

		for (j = 0; j < M;/*((M / 8) * 8);*/ j += 8) 
		{

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


	}

	return 1;
}

unsigned short int MVM_OMP()
{

	int i, j;
	
	#pragma omp parallel for private(i,j) shared(A1,X) reduction(+:Y)
	for (i = 0; i < M; i++) 
	{
		for (j = 0; j < M; j++) 
		{
			Y[i] += A1[i][j] * X[j];
		}
	}

	return 1;
} 

unsigned short int MVM_SSE()
{
__m128 num0, num1, num2, num3, num4, num5, num6;

	for (int i = 0; i < M; i++) {

		num3 = _mm_setzero_ps();
		for (int j = 0; j < M; j += 4) { 

			num0 = _mm_load_ps(&A1[i][j]);
			num1 = _mm_load_ps(&X[j]);
			num3 = _mm_fmadd_ps(num0, num1, num3);
		}

		num4 = _mm_hadd_ps(num3, num3);
		num4 = _mm_hadd_ps(num4, num4);

		_mm_store_ss(&Y[i], num4);
	}

	return 1;
}


unsigned short int MVM_SIMD()
{
	int i, j;
	
	#pragma omp parallel for private(i,j) shared(A1,X) reduction(+:Y)
	for (i = 0; i < M; i++) 
	{
		#pragma omp simd reduction(+:Y) aligned(Y,A1,X:64)
		for (j = 0; j < M; j++) 
		{
			Y[i] += A1[i][j] * X[j];
		}
	}

	return 1;
}

unsigned short int MVM_regBlock_2()
{
	double y0, y1, x0; 	
	
	for (int i = 0; i < M; i+=2) {
		y0 = Y[i];
		y1 = Y[i + 1];
		for (int j = 0; j < M; j++) {
			x0 = X[j];
			y0 += A1[i][j] * x0;
			y1 += A1[i+1][j] * x0;
		}
		Y[i] += y0;
		Y[i+1] +=y1;
	}

	return 1;
}

unsigned short int MVM_regBlock_8()
{
	double y0, y1, y2, y3, y4, y5, y6, y7, x0; 	
	
	for (int i = 0; i < M; i+=8) {
		y0 = Y[i];
		y1 = Y[i + 1];
		y2 = Y[i + 2];
		y3 = Y[i + 3];
		y4 = Y[i + 4];
		y5 = Y[i + 5];
		y6 = Y[i + 6];
		y7 = Y[i + 7];
		
		for (int j = 0; j < M; j++) {
			x0 = X[j];
			y0 += A1[i][j] * x0;
			y1 += A1[i+1][j] * x0;
			y2 += A1[i+2][j] * x0;
			y3 += A1[i+3][j] * x0;
			y4 += A1[i+4][j] * x0;
			y5 += A1[i+5][j] * x0;
			y6 += A1[i+6][j] * x0;
			y7 += A1[i+7][j] * x0;
		}
		Y[i] += y0;
		Y[i+1] +=y1;
		Y[i+2] +=y2;
		Y[i+3] +=y3;
		Y[i+4] +=y4;
		Y[i+5] +=y5;
		Y[i+6] +=y6;
		Y[i+7] +=y7;
	}

	return 1;
}

unsigned short int MVM_regBlock_13()
{
	double y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, x0; 	
	
	for (int i = 0; i < M; i+=13) {
		y0 = Y[i];
		y1 = Y[i + 1];
		y2 = Y[i + 2];
		y3 = Y[i + 3];
		y4 = Y[i + 4];
		y5 = Y[i + 5];
		y6 = Y[i + 6];
		y7 = Y[i + 7];
		y8 = Y[i + 8];
		y9 = Y[i + 9];
		y10 = Y[i + 10];
		y11 = Y[i + 11];
		y12 = Y[i + 12];
		
		for (int j = 0; j < M; j++) {
			x0 = X[j];
			y0 += A1[i][j] * x0;
			y1 += A1[i+1][j] * x0;
			y2 += A1[i+2][j] * x0;
			y3 += A1[i+3][j] * x0;
			y4 += A1[i+4][j] * x0;
			y5 += A1[i+5][j] * x0;
			y6 += A1[i+6][j] * x0;
			y7 += A1[i+7][j] * x0;
			y8 += A1[i+8][j] * x0;
			y9 += A1[i+9][j] * x0;
			y10 += A1[i+10][j] * x0;
			y11 += A1[i+11][j] * x0;
			y12 += A1[i+12][j] * x0;
		}
		Y[i] += y0;
		Y[i+1] +=y1;
		Y[i+2] +=y2;
		Y[i+3] +=y3;
		Y[i+4] +=y4;
		Y[i+5] +=y5;
		Y[i+6] +=y6;
		Y[i+7] +=y7;
		Y[i+8] +=y8;
		Y[i+9] +=y9;
		Y[i+10] +=y10;
		Y[i+11] +=y11;
		Y[i+12] +=y12;
	}

	return 1;
}

unsigned short int MVM_regBlock_16()
{
	double y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, x0; 	
	
	for (int i = 0; i < M; i+=16) {
		y0 = Y[i];
		y1 = Y[i + 1];
		y2 = Y[i + 2];
		y3 = Y[i + 3];
		y4 = Y[i + 4];
		y5 = Y[i + 5];
		y6 = Y[i + 6];
		y7 = Y[i + 7];
		y8 = Y[i + 8];
		y9 = Y[i + 9];
		y10 = Y[i + 10];
		y11 = Y[i + 11];
		y12 = Y[i + 12];
		y13 = Y[i + 13];
		y14 = Y[i + 14];
		y15 = Y[i + 15];
		
		for (int j = 0; j < M; j++) {
			x0 = X[j];
			y0 += A1[i][j] * x0;
			y1 += A1[i+1][j] * x0;
			y2 += A1[i+2][j] * x0;
			y3 += A1[i+3][j] * x0;
			y4 += A1[i+4][j] * x0;
			y5 += A1[i+5][j] * x0;
			y6 += A1[i+6][j] * x0;
			y7 += A1[i+7][j] * x0;
			y8 += A1[i+8][j] * x0;
			y9 += A1[i+9][j] * x0;
			y10 += A1[i+10][j] * x0;
			y11 += A1[i+11][j] * x0;
			y12 += A1[i+12][j] * x0;
			y13 += A1[i+13][j] * x0;
			y14 += A1[i+14][j] * x0;
			y15 += A1[i+15][j] * x0;
		}
		Y[i] += y0;
		Y[i+1] +=y1;
		Y[i+2] +=y2;
		Y[i+3] +=y3;
		Y[i+4] +=y4;
		Y[i+5] +=y5;
		Y[i+6] +=y6;
		Y[i+7] +=y7;
		Y[i+8] +=y8;
		Y[i+9] +=y9;
		Y[i+10] +=y10;
		Y[i+11] +=y11;
		Y[i+12] +=y12;
		Y[i+13] +=y13;
		Y[i+14] +=y14;
		Y[i+15] +=y15;
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
