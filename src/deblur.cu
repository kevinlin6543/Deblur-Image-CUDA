#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuComplex.h>
#include <cuda_profiler_api.h>
#include <cufft.h>

#define BLOCK_SIZE 32

// Kernel Functions for Deconvolution
__global__ void complexMul(cuComplex *A, cuComplex *B, cuComplex *C)
{
    unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
    C[i] = cuCmulf(A[i], B[i]);
}

__global__ void floatDiv(float *A, float *B, float *C)
{
    unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
    C[i] = A[i] / B[i];
}

__global__ void floatMul(float *A, float *B, float *C)
{
    unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
    C[i] = A[i] * B[i];
}

__global__ void convolution(float *A, float *B, float *C, int HA, int WA, int HB, int WB){
  int WC = WA - WB + 1;
  int HC = HA - HB + 1;
  __shared__ float tmp[BLOCK_SIZE][BLOCK_SIZE];

  int col = blockIdx.x * (BLOCK_SIZE - WB + 1) + threadIdx.x;
  int row = blockIdx.y * (BLOCK_SIZE - HB + 1) + threadIdx.y;
  int colI = col - WB + 1;
  int rowI = row - HB + 1;
  int sum = 0;

  if(rowI < HA && rowI >= 0 && colI < WA && colI >= 0)
    tmp[threadIdx.y][threadIdx.x] = A[colI * WA + rowI];
  else
    tmp[threadIdx.y][threadIdx.x] = 0;

  __syncthreads();

  if(threadIdx.y < (BLOCK_SIZE - HB + 1) && threadIdx.x < (BLOCK_SIZE - WB + 1) && row < (HC - HB + 1) && col < (WC - WB + 1)){
    for(int i = 0; i < HB; i++)
      for(int j = 0; j < WB; j++)
        sum += tmp[threadIdx.y + i][threadIdx.x + j] * B[j*WB + i]
    C[col*WC + row] = tmp;
  }

}

/* Parameters:
  nIter = Number of Iterations
  N1 = size of Dim 1
  N2 = size of Dim 2
  N3 = size of Dim 3
  *hImage = pointer to image memory
  *hPSF = pointer to PSF memory
  *hObject = pointer to output image memory
  PSF_x = num rows in PSF
  PSF_y = num cols in PSF
*/
int deconvLR(unsigned int nIter, size_t N1, size_t N2, size_t N3, float *hImage, float *hPSF, float *hObject, int PSF_x, int PSF_y){
  int ret = 0;
  cufftResult r;
  cudaError_t err;
  cufftHandle planR2C, planC2R;

  float *im = 0;
  float *obj = 0;
  float *psf = 0;
  float *otf = 0;
  void *buf = 0;
  void *tmp = 0;

  float psfFLIP[PSF_x][PSF_y];

  for(int i = 0; i < PSF_x; i++){
    for(int j = 0; j < PSF_y; i++){
      psfFLIP[PSF_x - i][PSF_y - j] = hPSF[i][j]
    }
  }

  size_t nSpatial = N1*N2*N3;
  size_t nFreq = N1*N2*(N3/2 + 1);
  size_t mSpatial;
  size_t mFreq;
  dim3 freqThreadsPerBlock, spatialThreadsPerBlock, freqBlocks, spatialBlocks;
  size_t tmpWork;
  err = numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = numBlocksThreads(nFreq, &freqBlocks, &freqThreadsPerBlock);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }

  mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(float);
  mFreq = freqBlocks.x * freqBlocks.y * freqBlocks.z * freqThreadsPerBlock.x * sizeof(cuComplex);

  cudaDeviceReset();
  cudaProfilerStart();
  // Memory Allocation
  err = cudaMalloc(&im, mSpatial)
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&obj, mSpatial)
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&psf, mSpatial)
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&otf, mSpatial)
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&buf, mSpatial)
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemset(im, 0, mSpatial);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemset(obj, 0, mSpatial);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }

  // Memory Copy for GPU Mem
  err = cudaMemcpy(im, hImage, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemcpy(psf, hPSF, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemcpy(otf, psfFLIP, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemcpy(obj, hImage, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  /*
  r = createPlans(N1, N2, N3, &planR2C, &planC2R, &tmp, &tmpWork);
  if(r){
    fprintf(stderr, "CuFFT error: %d\n", r);
    return r;
  }
  r = cufftExecR2C(planR2C, (float*)otf, otf);
  if(r){
    fprintf(stderr, "CuFFT error: %d\n", r);
    return r;
  }
  */
  for(int i = 0; i < nIter; i++){
    /*r = cufftExecR2C(planR2C, obj, (cufftComplex*)buf);
    if(r){
      fprintf(stderr, "CuFFT error: %d\n", r);
      return r;
    }
    ComplexMul<<<freqBlocks, freqThreadsPerBlock>>>((cuComplex*)buf, otf, (cuComplex*)buf);
    r = cufftExecC2R(planC2R, (cufftComplex*)buf, (float*)buf);
    if(r){
      fprintf(stderr, "CuFFT error: %d\n", r);
      return r;
    }*/
    convolution<<<spatialBlocks, spatialThreadsPerBlock>>>(obj, psf, buf, N1, N2, PSF_x, PSF_y)
    floatDiv<<<spatialBlocks, spatialThreadsPerBlock>>>(im, buf, buf);

    int HC = N1 - PSF_x + 1;
    int WC = N2 - PSF_y + 1;

    // FLIP PSF

    convolution<<<spatialBlocks, spatialThreadsPerBlock>>>(buf, otf, buf, HC, WC, PSF_x, PSF_y)
    /*r = cufftExecR2C(planR2C, (float*)buf, (cufftComplex*)buf);
    if(r){
      fprintf(stderr, "CuFFT error: %d\n", r);
      return r;
    }
    ComplexMul<<<freqBlocks, freqThreadsPerBlock>>>((cuComplex*)buf, otf, (cuComplex*)buf);
    r = cufftExecC2R(planC2R, (cufftComplex*)buf, (float*)buf);
    if(r){
      fprintf(stderr, "CuFFT error: %d\n", r);
      return r;
    }*/
    floatMul<<<spatialBlocks, spatialThreadsPerBlock>>>(buf, obj, obj);
  }
  // Copy output to host
  err = cudaMemcpy(hObject, obj, nSpatial*sizeof(float), cudaMemcpyDeviceToHost);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }

  ret = 0;
  // Clean Up Params and Return
  if(im)
    cudaFree(im);
  if(obj)
    cudaFree(obj);
  if(otf)
    cudaFree(otf);
  if(buf)
    cudaFree(buf);
  if(tmp)
    cudaFree(tmp);
  cudaProfilerStop();
  cudaDeviceReset();
  return ret;
}
