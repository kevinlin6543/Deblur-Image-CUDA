CXX=g++
CFLAGS = -Wall -Wextra -pedantic -ansi -O3
DEPS = ./lodepng/lodepng.cpp

all:
  $(CXX) process_image.cpp $(DEPS) $(CFLAGS) 

clean:
  rm *.exe *.o *.stackdump *~
