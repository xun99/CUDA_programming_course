////////////////////////////////////////////////////////////////////////
// GPU version of finding the average of az^2+bz+c 
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float a, b, c;



////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void average(float *d_z, float *d_result)
{
  int   ind;
  ind = threadIdx.x + blockIdx.x*blockDim.x;
  d_result[ind] = a*d_z[ind]*d_z[ind] + b*d_z[ind] + c;
}



////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
  int h_N=50000;
  float h_a, h_b, h_c, h_sum, h_ave;
  float *d_z, *h_result, *d_result;

   // initialise card
  findCudaDevice(argc, argv);


  // allocate memory on host and device
  h_result = (float *)malloc(sizeof(float)*h_N);

  checkCudaErrors( cudaMalloc((void **)&d_result, sizeof(float)*h_N) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*h_N) );


   // define constants and transfer to GPU
  h_a     = 4.0f;
  h_b     = 2.0f;
  h_c     = 1.0f;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(a,    &h_a,    sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b,    &h_b,    sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c,    &h_c,    sizeof(h_c)) );


  // random number generation
  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  checkCudaErrors( curandGenerateNormal(gen, d_z, h_N, 0.0f, 1.0f) );


  // execute kernel
  average<<<5, 100>>>(d_z, d_result);
  getLastCudaError("Average execution failed\n");


  // copy back results
  checkCudaErrors( cudaMemcpy(h_result, d_result, sizeof(float)*N,
                   cudaMemcpyDeviceToHost) );
  
  // compute average
  h_sum = 0.0;
  for (int i=0; i<h_N; i++) {
    h_sum += h_result[i];
  }

  h_ave = h_sum / (float)h_N;
  printf("\nThe average of az^2+bz+c is: %f.\n", h_ave);


  // Tidy up library
  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly
  free(h_result);
  checkCudaErrors( cudaFree(d_result) );
  checkCudaErrors( cudaFree(d_z) );

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();
}