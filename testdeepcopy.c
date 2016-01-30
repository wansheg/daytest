#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

struct mem_shape_desc;

typedef struct mem_shape_desc* (* get_mem_shape_desc_func  ) ( void* obj, int *cnt, uint16_t clause );

struct dim{
    __intptr_t offset;
    __intptr_t size;
};
struct mem_shape_desc{
  void**base;
  int element_size;
  unsigned short operation;
  unsigned short ndims;
  get_mem_shape_desc_func  deep_copy_shape;
  struct dim*    dims;
};



// function emitted by FE
//struct mem_shape_desc * BUILD_Vector1D_MemShapeDesc(void*, unsigned*, unsigned);

struct Vector1D {
	int n;
	double *data;
	#pragma acc shape include( data[0 : n ])
};

int main(void) 
{
    struct Vector1D vec;
	struct mem_shape_desc * L2 =  calloc(1, sizeof(struct mem_shape_desc));
        int i ;

        vec.n = 100;
        vec.data = (double * ) malloc( sizeof( double ) * 100 );
	#pragma acc data copy(vec)
#if 0 
        #pragma acc kernels loop 
#else 
        #pragma acc parallel loop 
#endif
        for ( i = 0; i <  100; i++ ) 
	{
          vec.data[i] = i;
	}
        for( i = 0; i < 100; i++ ) {
          printf( "%lf ", vec.data[i] );
        }
	return 0;
}

