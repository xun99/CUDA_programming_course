
__global__ void GPU_trid_2(int NX, int niter, float *u)
{
  extern __shared__  float a[], c[], d[];

  float aa, bb, cc, dd, bbi, lambda=1.0;
  int   tid;

  for (int iter=0; iter<niter; iter++) {

    // set tridiagonal coefficients and r.h.s.

    tid = threadIdx.x;
    actual_index = threadIdx.x + blockIdx.x*blockDim.x;

    bbi = 1.0f / (2.0f + lambda);
    
    if (tid>0)
      aa = -bbi;
    else
      aa = 0.0f;

    if (tid<blockDim.x-1)
      cc = -bbi;
    else
      cc = 0.0f;

    if (iter==0) 
      dd = lambda*u[actual_index]*bbi;
    else
      dd = lambda*dd*bbi;

    a[actual_index] = aa;
    c[actual_index] = cc;
    d[actual_index] = dd;

    // forward pass

    for (int nt=1; nt<NX; nt=2*nt) {
      __syncthreads();  // finish writes before reads

      bb = 1.0f;

      if (actual_index-nt >= 0) {
        dd = dd - aa*d[actual_index-nt];
        bb = bb - aa*c[actual_index-nt];
        aa =    - aa*a[actual_index-nt];
      }

      if (actual_index+nt < NX) {
        dd = dd - cc*d[actual_index+nt];
        bb = bb - cc*a[actual_index+nt];
        cc =    - cc*c[actual_index+nt];
      }

      __syncthreads();  // finish reads before writes


      bbi = 1.0f / bb;
      aa  = aa*bbi;
      cc  = cc*bbi;
      dd  = dd*bbi;

      a[actual_index] = aa;
      c[actual_index] = cc;
      d[actual_index] = dd;
    }
  }

  u[tid] = dd;
}

