CC = gcc
OPENCL_INCLUDE_PATH = /opt/AMDAPP/include
OPENCL_LIB_PATH = /opt/AMDAPP/lib/x86_64


canny: canny.c
	$(CC) -o canny src/util.c src/opencl_util.c canny.c -lm -O2 -ftree-vectorize -mavx2 -fopt-info-vec -ffast-math -fopenmp -I${OPENCL_INCLUDE_PATH} -L${OPENCL_LIB_PATH} -lOpenCL
