#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Funciones utiles y Clases  
#include "utils.h"
#include "utilsCuda.h"

//////////////////////////////////////////////////
// KERNEL
//////////////////////////////////////////////////
__global__ void sumaKernel(float *d_resultado, float *d_x, float *d_y, float *d_z, int numFilas){
	int i = (blockIdx.x*blockDim.x) + threadIdx.x; // indices de threads
	if (i < numFilas){
		d_resultado[i] = d_x[i] + d_y[i] + d_z[i];
	}
}

void suma_cu(int debug, int coutMode, float *h_resultado, float *h_x, float *h_y, float *h_z, int numFilas){
	cudaError_t error = cudaGetLastError();
	coutDeb(debug, "\n -> void suma_cu \n");

	coutMemoryUsage(debug, coutMode, "INICIAL"); // SE PRINTEA LA MEMORIA OCUPADA Y LIBRE

	//////////////////////////////////////////////////
	// MEMORY FROM HOST TO DEVICE
	//////////////////////////////////////////////////
	float *d_x;
	cudaMalloc((void **)&d_x, numFilas*sizeof(float));
	cudaMemcpy(d_x, h_x, numFilas*sizeof(float), cudaMemcpyHostToDevice);

	float *d_y;
	cudaMalloc((void **)&d_y, numFilas*sizeof(float));
	cudaMemcpy(d_y, h_y, numFilas*sizeof(float), cudaMemcpyHostToDevice);

	float *d_z;
	cudaMalloc((void **)&d_z, numFilas*sizeof(float));
	cudaMemcpy(d_z, h_z, numFilas*sizeof(float), cudaMemcpyHostToDevice);

	//////////////////////////////////////////////////
	// SE RESERVA MEMORIA PARA RESULTADOS
	//////////////////////////////////////////////////
	float *d_resultado;
	cudaMalloc((void **)&d_resultado, numFilas*sizeof(float));

	//////////////////////////////////////////////////
	// SE REVISA SI LA GPU EMITE ALGUN ERROR
	//////////////////////////////////////////////////
	cudaSyncGetLastError(error, "AL RESERVAR MEMORIA");

	coutMemoryUsage(debug, coutMode, "DESPUES DE RESERVA DE MEMORIA EN GPU"); // SE PRINTEA LA MEMORIA OCUPADA Y LIBRE

	// Se puede inicializar en algun entero, una variable de device
	cudaMemset(d_resultado, 0, numFilas*sizeof(float));

	//////////////////////////////////////////////////
	// SE CONFIGURA EL KERNEL
	//////////////////////////////////////////////////
	int threadsPerBlock = 1024; // 32 // 64 // 128 // 256 // 512 // 1024  (multiplos de 32, debido al warp size)
	int numBlocks = (numFilas + threadsPerBlock - 1) / threadsPerBlock;
	coutDeb(debug, "CUDA kernel launch with " + std::to_string(numBlocks) + " blocks of " + std::to_string(threadsPerBlock) + " threads.");

	//////////////////////////////////////////////////
	// SE INICIALIZA EL KERNEL
	//////////////////////////////////////////////////
	sumaKernel<<<numBlocks, threadsPerBlock >>>(d_resultado, d_x, d_y, d_z, numFilas);

	cudaSyncGetLastError(error, "AL EJECUTAR sumaKernel");

	//////////////////////////////////////////////////
	// Se copia memoria desde GPU a RAM
	//////////////////////////////////////////////////
	coutDeb(debug, "-> Memory from GPU to RAM ...");
	cudaMemcpy(h_resultado, d_resultado, numFilas*sizeof(float), cudaMemcpyDeviceToHost);

	cudaSyncGetLastError(error, "AL COPIAR MEMORIA");

	//////////////////////////////////////////////////
	// Se eliminan arreglos de GPU
	//////////////////////////////////////////////////
	coutDeb(debug, "-> Deleting GPU memory ...");
	checkCudaErrors(cudaFree(d_resultado));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_z));

	coutMemoryUsage(debug, coutMode, "FINAL"); // SE PRINTEA LA MEMORIA OCUPADA Y LIBRE
	coutDeb(debug, "end suma_cu");
}