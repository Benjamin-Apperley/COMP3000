#include "atax_original.h"
/* Matrix Transpose and Vector Multiplication */

/* Array initialization. */
float A[N][N], x[N], y[N], tmp[N], test[N];

int main()
{
	
	struct timespec start, end;
	uint64_t diff;
	double gflops;
	float out;

	init_array();
	
	clock_gettime(CLOCK_MONTONIC, &start);

	for(int i = 0; i < TIMES; i++)
	{
		kernel_atax();
	}

	clock_gettime(CLOCK_MONTONIC, &end);

	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	gflops = (double) ARITHMETICSL_OPS / (diff / TIMES);
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);
	printf("output = %f \n%f GigaFLOPS achieved\n", out, gflops);

	return 0;

}

void init_array ()
{
  int i, j;

  for (i = 0; i < N; i++)
      x[i] = i * M_PI;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = ((DATA_TYPE) i*(j+1)) / N;
}



/* Main kernels. */
void kernel_atax()
{
  int i, j;

#pragma scop
  	for (i = 0; i < N; i++)
	{
		y[i] = 0;
	}
    
  	for (i = 0; i < N; i++)
    	{
      		tmp[i] = 0;
      		for (j = 0; j < N; j++)
		{
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
      		for (j = 0; j < N; j++)
		{
			y[j] = y[j] + A[i][j] * tmp[i];
		}
    	}
#pragma endscop

}


/* compare correct output */
unsigned short int compare_atax()
{

	for (i = 0; i < N; i++)
	{
		test[i] = 0;
	}
    
  	for (i = 0; i < N; i++)
    	{
      		tmp[i] = 0;
      		for (j = 0; j < N; j++)
		{
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
      		for (j = 0; j < N; j++)
		{
			test[j] = test[j] + A[i][j] * tmp[i];
		}
    	}

	for (int i = 0; i < N; i++)
	{
		if (equal(Y[i], test[i]) == 1)
		{
			return 1;
		}
	}
	
	return 0;

}


