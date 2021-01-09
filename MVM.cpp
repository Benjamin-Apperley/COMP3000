#include "MVM.h"

int main()
{
	
	struct timespec start, end;
	uint64_t diff;
	double gflops;
	float out;
	int outcome;

	initialization_MVM();
	
	clock_gettime(CLOCK_MONOTONIC, &start);

	for(int i = 0; i < TIMES; i++)
	{
		MVM_default();
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	gflops = (double) ARITHMETICAL_OPS / (diff / TIMES);
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);
	printf("output = %f \n%f GigaFLOPS achieved\n", out, gflops);
	
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


unsigned short int MVM_default() {

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
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
