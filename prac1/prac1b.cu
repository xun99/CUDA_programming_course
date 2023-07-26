//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

// __global__ void my_first_kernel(float *x)
// {
//   int tid = threadIdx.x + blockDim.x*blockIdx.x;

//   x[tid] = (float) threadIdx.x;
// }

__global__ void vector_addition(float *d_A, float *d_B, float *d_C) 
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  d_C[tid] = d_B[tid] + d_A[tid];
}


//
// main code
//

int main(int argc, const char **argv)
{
  // float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n, i; 

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  int size_vector = 8;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  // nblocks  = 2;
  // nthreads = 8;
  // nsize    = nblocks*nthreads ;
  nblocks  = 2;
  nthreads = 4;


  // allocate memory for array

  // h_x = (float *)malloc(nsize*sizeof(float));
  // checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

  h_A = (float *)malloc(size_vector*sizeof(float));
  h_B = (float *)malloc(size_vector*sizeof(float));
  h_C = (float *)malloc(size_vector*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_A, size_vector*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_B, size_vector*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_C, size_vector*sizeof(float)));

  for(i=0; i<size_vector; i++) {
    h_A[i] = i + 1.0;
    h_B[i] = i + 1.0;
    h_C[i] = 0;
  }

  // cannot allocate memory for an array straightaway, need to do it one by one
  // h_A = {1, 2, 3, 4, 5, 6, 7, 8};
  // h_B = {9, 10, 11, 12, 13, 14, 15, 16};
  // h_C = {0, 0, 0, 0, 0, 0, 0, 0};

  // execute kernel
  
  // my_first_kernel<<<nblocks,nthreads>>>(d_x);
  checkCudaErrors( cudaMemcpy(d_A,h_A,size_vector*sizeof(float),
                 cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_B,h_B,size_vector*sizeof(float),
                 cudaMemcpyHostToDevice) );

  vector_addition<<<nblocks,nthreads>>>(d_A, d_B, d_C);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  // checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
  //                cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(h_C,d_C,size_vector*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  // for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);
  for (n=0; n<size_vector; n++) printf("%f \n",h_C[n]);

  // free memory 

  // checkCudaErrors(cudaFree(d_x));
  // free(h_x);
  checkCudaErrors(cudaFree(d_A));
  free(h_A);
  checkCudaErrors(cudaFree(d_B));
  free(h_B);
  checkCudaErrors(cudaFree(d_C));
  free(h_C);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
