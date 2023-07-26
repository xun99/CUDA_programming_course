

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[]; // this is local to the block!!!!!

    int tid = threadIdx.x;
    // int tid = threadIdx.x + blockIdx.x*blockDim.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[threadIdx.x + blockIdx.x*blockDim.x];

    // next, we perform binary tree reduction

    for (int d=blockDim.x/2; d>0; d=d/2) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[blockIdx.x] = temp[0]; //number of partial sums = num of blocks
    
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_elements, num_threads, mem_size, shared_mem_size;

  float *h_data, sum_CPU, sum_GPU;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_blocks   = 3;  // start with only 1 thread block
  num_threads = 512;
  num_elements = num_blocks*num_threads;
  // num_elements = 500;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  // if the num_element is not a power of 2
  // int new_num_elements = 1;
  // while(new_num_elements < num_elements){
  //   new_num_elements = 2*new_num_elements;
  // }
  // mem_size     = sizeof(float) * new_num_elements;

  h_data = (float*) malloc(mem_size);
  
  int i;
  for(i = 0; i < num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));
  
  // debugging
  // printf("\nThe values of initialisation for h_data are:\n");
  // for(int k=0; k<15; k++){
  //   printf("\n%f\n", h_data[k]);
  // }

  // if the num_element is not a power of 2
  // for(int j = i; j<new_num_elements; j++)
  //   h_data[j] = 0.0f;

  // compute reference solution

  sum_CPU = reduction_gold(h_data, num_elements);
  // printf("\nThe result of CPU is: %f\n", sum_CPU);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float) * num_blocks) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, sizeof(float) * num_blocks,
                              cudaMemcpyDeviceToHost) );

  // check results
  sum_GPU = 0.0;
  for(int k=0; k<num_blocks; k++){
    // printf("\nThe result of GPU is:\n");
    // printf("\n%f\n", h_data[k]);
    sum_GPU += h_data[k];
  }
  printf("After global reduction, reduction error = %f\n",sum_GPU-sum_CPU);

  // cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
