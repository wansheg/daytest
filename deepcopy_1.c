#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// function emitted by FE
//struct mem_shape_desc * BUILD_Vector1D_MemShapeDesc(void*, unsigned*, unsigned);

struct Vector1D {
  int n;
  double *data;
  #pragma acc shape include( data[0 : n ])
};

struct Matrix{
  int rows;
  struct Vector1D* vecs;
  #pragma acc shape include( vecs[0 : rows ])
};

int main(void) 
{
    struct Matrix mat;
    int i,j ;

    mat.rows = 10;
    mat.vecs = malloc( sizeof( struct Vector1D ) * 10 ) ;
    for ( i = 0; i < 10; i++ ) {
      mat.vecs[i].n = 10;
      mat.vecs[i].data = (double * ) malloc( sizeof( double ) * 10 );
    }
	#pragma acc data copy(mat)
#if 0 
    #pragma acc kernels loop 
#else 
    #pragma acc parallel loop 
#endif
    for ( i = 0; i <  10; i++ ) 
	{
      for ( j = 0; j < 10; j++ ) {
          mat.vecs[i].data[j] = i * j;
      }
	}
    for( i = 0; i < 10; i++ ) {
      for ( j = 0; j < 10; j++ )
        printf( "%lf ", mat.vecs[i].data[j] );
    }
	return 0;
}

