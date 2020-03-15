CXX = g++
CFLAGS = -Wall -Wextra -pedantic -ansi -O3
DEPS = ./lodepng/lodepng.cpp ./metrics.cpp
CPU_CODE = cpu_deblur.cpp

all:
	$(CXX) process_image.cpp $(DEPS) $(CFLAGS)

cpu:
	$(CXX) $(CPU_CODE) -o cpu_deblur.out `pkg-config --cflags --libs opencv`

clean:
	rm *.exe *.o *.stackdump *~
