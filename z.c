// support AVX512 
// the  "int64" to "double" SIMD CVT
#include <stdio.h>
#include <math.h>
#define N 1000000000000LL

int main()
{   double pi = 0.0; long i;
      #pragma omp parallel for reduction(+:pi)
      for (i=0; i<N; i++)
      {
          double t = (double)((i+0.5)/N); pi += 4.0/(1.0+t*t);
      }
      printf("pi = %f\n",pi/N);            
      return 0;
}
