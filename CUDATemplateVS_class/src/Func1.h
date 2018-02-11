#ifndef FUNC1_H
#define FUNC1_H

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

void suma_cu(
	int debug,
	int coutMode,
	float *h_resultado,
	float *h_x,
	float *h_y,
	float *h_z,
	int numFilas
	);

class Func1{
public:
	Func1(
		int debug,
		int coutMode,
		int outputCSV,
		int numFilas,
		std::string fileOut
	);

private:
	int func1Main(
		int debug,
		int coutMode,
		int outputCSV,
		int numFilas,
		std::string fileOut
	);
};

#endif