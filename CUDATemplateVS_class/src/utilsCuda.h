#ifndef UTILSCUDA_H
#define UTILSCUDA_H

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <algorithm>

// Required to include CUDA vector types
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

void cudaSyncGetLastError(cudaError_t error, std::string message);
void coutMemoryUsage(int debug, int coutMode, std::string stage);

#endif