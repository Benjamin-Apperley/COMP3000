#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "atax.h"


/* Array initialization. */
static
void init_array (int nx, int ny, A[][], x[])
{
  int i, j;

  for (i = 0; i < ny; i++)
      x[i] = i * M_PI;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
}




void kernel_atax(int nx, int ny,A[][], x[], y[], tmp[])
{
  int i, j;

#pragma scop
  	for (i = 0; i < ny; i++)
	{
		y[i] = 0;
	}
    
  	for (i = 0; i < nx; i++)
    	{
      		tmp[i] = 0;
      		for (j = 0; j < ny; j++)
		{
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
      		for (j = 0; j < _PB_NY; j++)
		{
			y[j] = y[j] + A[i][j] * tmp[i];
		}
    	}
#pragma endscop

}


