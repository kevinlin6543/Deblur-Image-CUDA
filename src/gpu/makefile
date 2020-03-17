#ARCH = aarch64
NVCC = /usr/local/cuda-10.0/bin/nvcc
NVCC_LIB = /usr/local/cuda-10.0/lib64
INCLUDES = -I/usr/local/cuda-10.0/samples/common/inc

deblur: deblur.o metrics.o lodepng.o 
	g++ -o deblur -L$(NVCC_LIB) -lcuda -lcudart deblur.o metrics.o lodepng.o

deblur.o: deblur.cu 
	$(NVCC) $(INCLUDES) -c deblur.cu

metrics.o: ../metrics/metrics.cpp ../metrics/metrics.hpp
	g++ -c ../metrics/metrics.cpp 

lodepng.o: ../lodepng/lodepng.cpp ../lodepng/lodepng.h
	g++ -c ../lodepng/lodepng.cpp -Wall -Wextra -pedantic -ansi -O3
