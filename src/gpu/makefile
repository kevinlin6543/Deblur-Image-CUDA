#ARCH = aarch64
NVCC = /usr/local/cuda-10.0/bin/nvcc
NVCC_LIB = /usr/local/cuda-10.0/lib64
#INCLUDES = /usr/local/cuda-10.0/samples/common/inc
INCLUDES = /user/local/cuda-10.0/include
#CUDA_LIB = 
CUDART_LIB = /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so

deblur: deblur.o metrics.o lodepng.o
	g++ -o deblur deblur.o -L$(NVCC_LIB) -lcudart metrics.o lodepng.o

deblur.o: deblur.cu 
	$(NVCC) -I. -I$(INCLUDES) -c deblur.cu

metrics.o: ../metrics/metrics.cpp ../metrics/metrics.hpp
	g++ -c ../metrics/metrics.cpp 

lodepng.o: ../lodepng/lodepng.cpp ../lodepng/lodepng.h
	g++ -c ../lodepng/lodepng.cpp -Wall -Wextra -pedantic -ansi -O3
