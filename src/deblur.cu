#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuComplex.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include "./lodepng/lodepng.h"
#include "./metrics.hpp"
#include "./gpu_time.hpp"
#include <iostream>

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

// Assumes RGB (3-Channels)
/* Parameters:
  *A = input matrix 1
  *B = input matrix 2
  *C = output matrix
  HA = height of Matrix A
  WA = width of Matrix A
  HB = height of Matrix B
  WB = width of Matrix B
*/
__global__ void convolution(float *A, float *B, float *C, int HA, int WA, int HB, int WB){
  int WC = WA - WB + 1;
  int HC = HA - HB + 1;
  __shared__ float tmp[BLOCK_SIZE][BLOCK_SIZE][3];

  int col = blockIdx.x * (BLOCK_SIZE - WB + 1) + threadIdx.x;
  int row = blockIdx.y * (BLOCK_SIZE - HB + 1) + threadIdx.y;
  int colI = col - WB + 1;
  int rowI = row - HB + 1;
  int sum = 0;

  if(rowI < HA && rowI >= 0 && colI < WA && colI >= 0){
    tmp[threadIdx.y][threadIdx.x][0] = A[colI * WA + rowI + 0*(HA*WA)];
    tmp[threadIdx.y][threadIdx.x][1] = A[colI * WA + rowI + 1*(HA*WA)];
    tmp[threadIdx.y][threadIdx.x][2] = A[colI * WA + rowI + 2*(HA*WA)];
  }
  else{
    tmp[threadIdx.y][threadIdx.x][0] = 0;
    tmp[threadIdx.y][threadIdx.x][1] = 0;
    tmp[threadIdx.y][threadIdx.x][2] = 0;
  }


  __syncthreads();

  if(threadIdx.y < (BLOCK_SIZE - HB + 1) && threadIdx.x < (BLOCK_SIZE - WB + 1) && row < (HC - HB + 1) && col < (WC - WB + 1)){
    for(int i = 0; i < HB; i++){
      for(int j = 0; j < WB; j++){
        sum0 += tmp[threadIdx.y + i][threadIdx.x + j][0] * B[j*WB + i + 0*(HB*WB)];
        sum1 += tmp[threadIdx.y + i][threadIdx.x + j][1] * B[j*WB + i + 1*(HB*WB)];
        sum2 += tmp[threadIdx.y + i][threadIdx.x + j][2] * B[j*WB + i + 2*(HB*WB)];
      }
    }
    C[col*WC + row + 0*(HC*WC)] = sum0;
    C[col*WC + row + 1*(HC*WC)] = sum1;
    C[col*WC + row + 2*(HC*WC)] = sum2;
  }

}

/* Parameters:
  nIter = Number of Iterations
  N1 = size of Dim 1  (im.x)
  N2 = size of Dim 2  (im.y)
  N3 = size of Dim 3  (im.channels) (3 for RGB)
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


  // FLIP PSF for algorithm operation
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


/* Decode the PNG into a 1D vector */
std::vector<int> decodePNG(const char* filename, unsigned &w, unsigned &h) {
    std::vector<unsigned char> image;

    lodepng::decode(image, w, h, filename);

    std::vector<int> image_without_alpha;
    for(unsigned int i = 0; i < image.size(); i++) {
        if (i % 4 != 3) {
            image_without_alpha.push_back((int)image[i]);
        }
    }

    return image_without_alpha;
}

/* Copy contents of vector to arr to be used in CUDA kernel functions */
float * vecToArr(std::vector<int> image)
{
  float *arr = malloc(image.size() * sizeof(float));
  if(!arr)
  {
    std::cerr << "Error converting vector to array" << std::endl;
    exit(-1);
  }

  std::copy(image.begin(), image.end(), arr);

  return arr;
}


int main(int argc, char **argv)
{
  if(argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " blurry.png ref.png" << std::endl;
    exit(-1);
  }

  /* Convert the PNGs to 1D vectors */
  unsigned w_blurry, h_blurry;
  unsigned w_ref, h_ref;
  std::vector<int> blurry = decodePNG(argv[1], w_blurry, h_blurry);
  std::vector<int> ref = decodePNG(argv[2], w_ref, h_ref);

  /* Convert image into array to be used in kernel functions */
  float *blurry_arr = vecToArr(blurry);

  /* Create timing class */
  gpu_time gt;

  gt.begin();  
  /* Call kernel function with blurry_arr, w_blurry, h_blurry */
  /* FUNCTION */
  gt.end();

  /* Re-convert back to vector for metrics computation */
  std::vector<int> out_vec;
  out_vec.insert(out_vec.begin(), std::begin(out_arr), std::end(out_arr));

  /* Metrics */
  std::cout << "Elapsed time: " << gt.elap_time() << std::endl;
  std::cout << "MSE: " << _mse(out_vec, w_blurry, h_blurry, ref) << std::endl;
  std::cout << "pSNR: " << psnr(out_vec, w_blurry, h_blurry, ref) << std::endl;

  /* TODO: Append alpha values to out_vec in order to see the deblurred image */

  return 0;
}


