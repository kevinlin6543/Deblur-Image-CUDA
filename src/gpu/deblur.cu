#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuComplex.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include "../lodepng/lodepng.h"
#include "../metrics/metrics.hpp"
//#include "./gpu_time.hpp"
#include <iostream>
#include <array>
#include <vector>

#define BLOCK_SIZE 32

using namespace std;

// Kernel Functions for Deconvolution
__global__ void ComplexMul(cuComplex *A, cuComplex *B, cuComplex *C)
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

  int col = blockIdx.x * (BLOCK_SIZE - WB + 1) + threadIdx.x + ((WA-WC)/2);
  int row = blockIdx.y * (BLOCK_SIZE - HB + 1) + threadIdx.y + ((HA-HC)/2);
  int colI = col - WB + 1;
  int rowI = row - HB + 1;
  int sum = 0;

  if(rowI < HA && rowI >= 0 && colI < WA && colI >= 0){
    tmp[threadIdx.y][threadIdx.x][0] = A[colI + rowI*WA + 0*(HA*WA)];
    tmp[threadIdx.y][threadIdx.x][1] = A[colI + rowI*WA + 1*(HA*WA)];
    tmp[threadIdx.y][threadIdx.x][2] = A[colI + rowI*WA + 2*(HA*WA)];
  }
  else{
    tmp[threadIdx.y][threadIdx.x][0] = 0;
    tmp[threadIdx.y][threadIdx.x][1] = 0;
    tmp[threadIdx.y][threadIdx.x][2] = 0;
  }


  __syncthreads();
  float sum0 = 0, sum1 = 0, sum2 = 0;
  if(threadIdx.y < (BLOCK_SIZE - HB + 1) && threadIdx.x < (BLOCK_SIZE - WB + 1) && row < (HC - HB + 1) && col < (WC - WB + 1)){
    for(int i = 0; i < HB; i++){
      for(int j = 0; j < WB; j++){
        sum0 += tmp[threadIdx.y + i][threadIdx.x + j][0] * B[j + i*WB + 0*(HB*WB)];
        sum1 += tmp[threadIdx.y + i][threadIdx.x + j][1] * B[j + i*WB + 1*(HB*WB)];
        sum2 += tmp[threadIdx.y + i][threadIdx.x + j][2] * B[j + i*WB + 2*(HB*WB)];
      }
    }
    C[col + row*WA + 0*(HA*WA)] = sum0;
    C[col + row*WA + 1*(HA*WA)] = sum1;
    C[col + row*WA + 2*(HA*WA)] = sum2;
  }

}

static cudaError_t numBlocksThreads(unsigned int N, dim3 *numBlocks, dim3 *threadsPerBlock) {
    unsigned int BLOCKSIZE = 128;
    int Nx, Ny, Nz;
    int device;
    cudaError_t err;
    if(N < BLOCKSIZE) {
        numBlocks->x = 1;
        numBlocks->y = 1;
        numBlocks->z = 1;
        threadsPerBlock->x = N;
        threadsPerBlock->y = 1;
        threadsPerBlock->z = 1;
        return cudaSuccess;
    }
    threadsPerBlock->x = BLOCKSIZE;
    threadsPerBlock->y = 1;
    threadsPerBlock->z = 1;
    err = cudaGetDevice(&device);
    if(err)
      return err;
    err = cudaDeviceGetAttribute(&Nx, cudaDevAttrMaxBlockDimX, device);
    if(err)
      return err;
    err = cudaDeviceGetAttribute(&Ny, cudaDevAttrMaxBlockDimY, device);
    if(err)
      return err;
    err = cudaDeviceGetAttribute(&Nz, cudaDevAttrMaxBlockDimZ, device);
    if(err)
      return err;
    printf("Nx: %d, Ny: %d, Nz: %d\n", Nx, Ny, Nz);
    unsigned int n = (N-1) / BLOCKSIZE + 1;
    unsigned int x = (n-1) / (Ny*Nz) + 1;
    unsigned int y = (n-1) / (x*Nz) + 1;
    unsigned int z = (n-1) / (x*y) + 1;
    if(x > Nx || y > Ny || z > Nz) {
        return cudaErrorInvalidConfiguration;
    }
    numBlocks->x = x;
    numBlocks->y = y;
    numBlocks->z = z;

    return cudaSuccess;
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
int deconvLR(unsigned int nIter, size_t N1, size_t N2, size_t N3, float *hImage, float *hPSF, float *hObject, int PSF_x, int PSF_y, float *hPSF2){
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

  //cerr << PSF_x << ":" << PSF_y << endl;
  // FLIP PSF for algorithm operation
  /*float *psfFLIP = (float *)malloc(PSF_x * PSF_y * 3 * sizeof(float));
  cerr << "Got Into Function" << endl;
  for(int i = 0; i < PSF_x; i++){
    for(int j = 0; j < PSF_y; j++){
      //cerr << i << " : " << j << endl;
      psfFLIP[(PSF_x*PSF_y) - j - (i*PSF_y) - 1 + 0*(PSF_x*PSF_y)] = hPSF[j + PSF_y*i + 0*(PSF_x*PSF_y)];
      psfFLIP[(PSF_x*PSF_y) - j - (i*PSF_y) - 1 + 1*(PSF_x*PSF_y)] = hPSF[j + PSF_y*i + 1*(PSF_x*PSF_y)];
      psfFLIP[(PSF_x*PSF_y) - j - (i*PSF_y) - 1 + 2*(PSF_x*PSF_y)] = hPSF[j + PSF_y*i + 2*(PSF_x*PSF_y)];
    }
  }

  for(int i = 0; i < PSF_x*PSF_y; i++){
    cerr << "PSFR: " << hPSF[i] << " FLIP:" << psfFLIP[i] << endl;
    cerr << "PSFG: " << hPSF[i + 1*(PSF_x*PSF_y)] << " FLIP:" << psfFLIP[i + 1*(PSF_x*PSF_y)] << endl;
    cerr << "PSFB: " << hPSF[i + 2*(PSF_x*PSF_y)] << " FLIP:" << psfFLIP[i + 2*(PSF_x*PSF_y)] << endl;

  }
  cerr << "Survived Flip" << endl;*/

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

  cerr << "Survived Setup" << endl;
  // Memory Allocation
  err = cudaMalloc(&im, mSpatial);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&obj, mSpatial);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&psf, mSpatial);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&otf, mSpatial);
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMalloc(&buf, mSpatial);
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
  cerr << "Survived Malloc" << endl;
  // Memory Copy for GPU Mem
  err = cudaMemcpy(im, hImage, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  cerr << "1st MEMCPY" << endl;
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemcpy(psf, hPSF, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  cerr << "2nd MEMCPY" << endl;
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemcpy(otf, hPSF2, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  cerr << "3rd MEMCPY" << endl;
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  err = cudaMemcpy(obj, hImage, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
  cerr << "4th MEMCPY" << endl;
  if(err){
    fprintf(stderr, "CUDA error: %d\n", err);
    return err;
  }
  cerr << "Survived Memcpy" << endl;
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
    // cerr << "Iteration: " << i << endl;
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
    // cerr << N1 << " , " << N2 << " , " << N3 << endl;
    convolution<<<spatialBlocks, spatialThreadsPerBlock>>>(obj, psf, (float *)buf, N1, N2, PSF_x, PSF_y);
    floatDiv<<<spatialBlocks, spatialThreadsPerBlock>>>(im, (float *)buf, (float *)buf);

    //int HC = N1 - PSF_x + 1;
    //int WC = N2 - PSF_y + 1;

    convolution<<<spatialBlocks, spatialThreadsPerBlock>>>((float *)buf, otf, (float *)buf, N1, N2, PSF_x, PSF_y);
    //cerr << (
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
    floatMul<<<spatialBlocks, spatialThreadsPerBlock>>>((float *)buf, obj, obj);
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

int encodePNG(vector<int> im, unsigned &w, unsigned &h){
    int counter = 0;
    vector<unsigned char> image;
    for(int i = 0; i < im.size(); i++){
        if(i%3 == 0 && i != 0){
            image.push_back(255);
        }
        image.push_back(im[i]);
    }
    image.push_back(255);
    cerr << "SIZE: " << image.size() <<endl;
    unsigned err = lodepng::encode((const char *)"./img/test3.png", image, w, h);
    cerr << lodepng_error_text(err) << endl;
    //lodepng::save_file
    return 0;
}

/* Copy contents of vector to arr to be used in CUDA kernel functions */
float * vecToArr(std::vector<int> image)
{
  float *arr = (float *)malloc(image.size() * sizeof(float));
  if(!arr)
  {
    std::cerr << "Error converting vector to array" << std::endl;
    exit(-1);
  }

  std::copy(image.begin(), image.end(), arr);

  return arr;
}

const double pi = 3.14159265358979323846;
std::vector<std::vector<std::vector<double> > > calculatePSF(std::vector<std::vector<std::vector<double> > > &psf_hat) {
	int psf_size = 5;
	double mean_row = 0.0;
	double mean_col = 0.0;

	double sigma_row = 12.0;
	double sigma_col = 6.0;

	double sum = 0.0;
	double temp;

	std::vector<std::vector<double> > psf_init(psf_size, std::vector<double> (psf_size));
	std::vector<std::vector<std::vector<double> > > psf_final(psf_size, std::vector<std::vector<double> > (psf_size, std::vector<double> (3)));
	psf_hat.resize(psf_size, std::vector<std::vector<double> >(psf_size, std::vector<double>(3)));

	for (unsigned j = 0; j< psf_init.size(); j++) {
		for (unsigned k = 0; k< psf_init[0].size(); k++) {
			temp = exp(-0.5 * (pow((j - mean_row) / sigma_row, 2.0) + pow((k - mean_col) / sigma_col, 2.0))) / (2* pi * sigma_row * sigma_col);
			sum += temp;
			psf_init[j][k] = temp;
		}
	}

	for (unsigned row = 0; row<psf_init.size(); row++) {
		for (unsigned col = 0; col<psf_init[0].size(); col++) {
			psf_init[row][col] /= sum;
		}
	}

	for (unsigned row = 0; row<psf_init.size(); row++) {
		for (unsigned col = 0; col<psf_init[0].size(); col++) {
			//std::cerr << "[" << row << ", " << col << "] = " << psf_init[row][col] << '\n';
			double curr = psf_init[row][col];
			psf_final[row][col][0] = curr;
			psf_final[row][col][1] = curr;
			psf_final[row][col][2] = curr;
		}
	}
	for (int row = 0; row < psf_size; row++) {
		for (int col = 0; col < psf_size; col++) {
			int y = psf_size - 1 - row;
			int x = psf_size - 1 - col;
			psf_hat[y][x][0] = psf_final[row][col][0];
			psf_hat[y][x][1] = psf_final[row][col][1];
			psf_hat[y][x][2] = psf_final[row][col][2];
		}
	}
	cerr << psf_final[0][0][0] << endl;
	return psf_final;
}

void flip(float *hPSF, float *psfFLIP, int PSF_x, int PSF_y){
  for(int i = 0; i < PSF_x; i++){
    for(int j = 0; j < PSF_y; j++){
      psfFLIP[(PSF_x*PSF_y) - j - (i*PSF_y) - 1 + 0*(PSF_x*PSF_y)] = hPSF[j + PSF_y*i + 0*(PSF_x*PSF_y)];
      psfFLIP[(PSF_x*PSF_y) - j - (i*PSF_y) - 1 + 1*(PSF_x*PSF_y)] = hPSF[j + PSF_y*i + 1*(PSF_x*PSF_y)];
      psfFLIP[(PSF_x*PSF_y) - j - (i*PSF_y) - 1 + 2*(PSF_x*PSF_y)] = hPSF[j + PSF_y*i + 2*(PSF_x*PSF_y)];
    }
  }
}

void convert1D(std::vector<std::vector<std::vector<double> > > &a, std::vector<double> &vec1D) {
	for(unsigned i = 0; i < a.size(); i++) {
		for(unsigned j = 0; j < a[0].size(); j++) {
			vec1D.push_back(a[i][j][0]);
			vec1D.push_back(a[i][j][1]);
			vec1D.push_back(a[i][j][2]);
			//vec1D.push_back((unsigned char)(255));
		}
	}
}


float * vecToArr2(std::vector<double> image)
{
  float *arr = (float *)malloc(image.size() * sizeof(float));
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
    std::cerr << "Usage:  " << argv[0] << " blurry.png ref.png" << std::endl;
    exit(-1);
  }

  int ret = 0;
  int im_z = 3;
  int nIter = 1;
  int PSF_x = 5;
  int PSF_y = 5;

  std::vector<std::vector<std::vector<double>>> psf_hat;
  std::vector<std::vector<std::vector<double>>> psf_vec = calculatePSF(psf_hat);
  std::vector<double> psf_1d;
  convert1D(psf_vec, psf_1d);
  float *PSF = vecToArr2(psf_1d);
  float *PSF2 = vecToArr2(psf_1d);
  flip(PSF, PSF2, 5, 5);

  /* Convert the PNGs to 1D vectors */
  unsigned w_blurry, h_blurry;
  unsigned w_ref, h_ref;
  std::vector<int> blurry = decodePNG(argv[1], w_blurry, h_blurry);
  std::vector<int> ref = decodePNG(argv[2], w_ref, h_ref);

  /* Convert image into array to be used in kernel functions */
  float *blurry_arr = vecToArr(blurry);
  float *out_arr = (float *)malloc(w_blurry * h_blurry * im_z *sizeof(float));

  /* Create timing class */
  //gpu_time gt;
  //gt.begin();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  std::cerr << "Survived Initialization" << std::endl;
  /* Call kernel function with blurry_arr, w_blurry, h_blurry */
  ret = deconvLR(nIter, h_blurry, w_blurry, im_z, blurry_arr, PSF, out_arr, PSF_x, PSF_y, PSF2);
  std::cerr << "Survived Algorithm" << std::endl;
  cudaEventRecord(stop, 0);
  float t = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t, start, stop);
  //gt.end();

  /* Re-convert back to vector for metrics computation */
  std::vector<int> out_vec;
  //out_vec.insert(out_vec.begin(), std::begin(out_arr), std::end(out_arr));
  for(int i = 0; i < 2764800; i++)
    out_vec.push_back( static_cast<int>(out_arr[i]) );
  cerr << "Size of output vector: " << out_vec.size() << endl;
  cerr << "Size of ref vector: " << ref.size() << endl;
  /* Metrics */
  std::cout << "Elapsed time: " << t << std::endl;
  std::cout << "MSE: " << _mse(out_vec, w_blurry, h_blurry, ref) << std::endl;
  std::cout << "pSNR: " << psnr(out_vec, w_blurry, h_blurry, ref) << std::endl;
  int test = encodePNG(out_vec, w_blurry, h_blurry);
  /* TODO: Append alpha values to out_vec and change back to char in order to see the deblurred image THIS TODO IS DONE*/

  return 0;
}
