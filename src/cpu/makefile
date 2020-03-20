CXX = g++
CFLAGS = -Wall -Wextra -pedantic -ansi -O3 -std=c++0x
DEPS = ../lodepng/lodepng.cpp ../metrics/metrics.cpp
CPU_CODE = cpu_deblur.cpp

all:
	$(CXX) process_image.cpp $(DEPS) $(CFLAGS)

cpu:
	$(CXX) $(CPU_CODE) -o cpu_deblur.out $(DEPS) $(CFLAGS)

clean:
	rm *.exe *.o *.stackdump *~
