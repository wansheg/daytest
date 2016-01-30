/*
The example should be supportable with shape and policy directives, like this:
However, if we stick to the paper, the copy clause of the containers


has to be modified to use the desired copy policy (in our out) on the
"internal" member, and I'm not sure if the paper already specifies how
that should look like.

because my_policy's type is not that of containers.

like it would not include any additional members of the superclass
(that are not present in this example).

Perhaps we want to modify the specification to allow the desired
behaviour without changing the data clause, like having a default
policy apply.

Additional note
--------
Actually, I just noticed we can specify a policy for container_t that
references a policy for the internal member.


But the data clauses still need to be modified to use foo:
*/

typedef struct internal_s {
   int len;
   float *array;
   float *result;
#pragma acc shape  include(array[0:len],result[0:len]) init_needed(len)
#pragma acc policy(my_policy)  copyin(array) copyout(result)
} internal_t;



typedef struct container_s {
   internal_t internal;
   float *array_ptr;
#pragma acc shape include( internal, array_ptr[@internal] )
} container_t;

container_t *containers;

int container_length = 10;
int n = 100000;

int main()
{
  int i , j, k;
  container_t * containers = malloc( sizeof( container_t ) * container_length );
  for( i = 0; i < container_length; i++ ) {
    float* data = malloc( sizeof( float ) * n );
    for( j = 0; j < n; j++ ) {
      data[j] = j* (float) n + i ;
    }
    containers[i].internal.len = n;
    containers[i].internal.array = malloc( sizeof( float ) * n );
    containers[i].internal.result = malloc( sizeof( float ) * n );
    containers[i].array_ptr = containers[i].internal.result;
    for ( j = 0; j < n; j++ ) {
      containers[i].internal.array[j] = data[j];
      containers[i].array_ptr[j] = data[j] - 11;
    }
  }

  #pragma acc data copy( containers[0:container_length] )
  for ( i = 0; i < container_length; i++ ) {
    #pragma acc parallel loop
    for ( j = 0; j < containers[i].internal.len; j++ ) {
      containers[i].internal.result[j] = containers[i].internal.array[j] * 2;
    }
  }
}



//#pragma acc data invoke<my_policy>(containers[0:n]) might be illegal
//#pragma acc data invoke<my_policy>(containers[0:n].internal) looks
//#pragma acc data copy(containers[0:n])
//#pragma acc policy(foo) type(container_t) invoke<my_policy>(internal).
//#pragma acc data invoke<foo>(containers[0:n])
