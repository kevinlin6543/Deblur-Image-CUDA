# Project 1 for Advance Computer Architecture
Deblurs image based on Lucy Richardson Algorithm

## Requirements
```
cuda 10 toolkit
gcc
git
```

## Usage
1. Run on cross-compile machine
```bash
git clone https://github.com/kevinlin6543/Deblur-Image-CUDA.git
cd ./Deblur-Image-CUDA/src
make
```
2. Results in a 'deblur' file. Move file to Jetson Nano or equivalent

3. On Jetson Nano:
```bash
cd [Directory with deblur file]
./deblur [blurry png file] [original png file] [number of iterations] [output png file]
```

## Usage of CPU only code
```bash
git clone https://github.com/kevinlin6543/Deblur-Image-CUDA.git
cd ./Deblur-Image-CUDA/src
make cpu_deblur
./cpu_deblur.out [blurry png file] [original png file] [number of iterations] [output png file]
```
