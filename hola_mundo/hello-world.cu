// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA
// with an array of offsets. Then the offsets are added in parallel
// to produce the string "World!"
// By Ingemar Ragnemalm 2010

// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ void kernel_hello(char *a, int *b){
	a[threadIdx.x] += b[threadIdx.x];
}

int main(){

	char a[N] = "Hello \0\0\0\0\0\0";
	int  b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	char *a_device; 
	int  *b_device;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

	cudaMalloc( (void**)&a_device, csize ); 
	cudaMalloc( (void**)&b_device, isize ); 

	cudaMemcpy( a_device, a, csize, cudaMemcpyHostToDevice ); 
	cudaMemcpy( b_device, b, isize, cudaMemcpyHostToDevice ); 
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );

	kernel_hello<<<dimGrid, dimBlock>>>(a_device, b_device);

	cudaMemcpy( a, a_device, csize, cudaMemcpyDeviceToHost ); 
	cudaFree( a_device );
	cudaFree( b_device );
	
	printf("%s\n", a);

	return EXIT_SUCCESS;
}