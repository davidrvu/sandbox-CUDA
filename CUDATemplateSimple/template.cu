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

//////////////////////////////////////////////////
// KERNEL
//////////////////////////////////////////////////
__global__ void sumaKernel(float *d_resultado, float *d_x, float *d_y, float *d_z, int numFilas){
	int i = (blockIdx.x*blockDim.x) + threadIdx.x; // indices de threads
	if (i < numFilas){
		d_resultado[i] = d_x[i] + d_y[i] + d_z[i];
	}
}

//////////////////////////////////////////////////
// FUNCIONES AUXILIARES
//////////////////////////////////////////////////

void coutMemoryUsage(int debug, int coutMode, std::string stage){ // Muestra el uso de la memoria global de la GPU
	if (debug >= 1){
		size_t free_byte;
		size_t total_byte;
		cudaMemGetInfo(&free_byte, &total_byte);
		float free_db = (float)free_byte / 1024.0f / 1024.0f;
		float total_db = (float)total_byte / 1024.0f / 1024.0f;
		float used_db = total_db - free_db;
		if(coutMode == 0){
			std::cout << "__________________________________________________________________________________________________" << std::endl;
			std::cout << "    GPU memory usage in stage " << stage << ": used = " << used_db << "[MB], free = " << free_db << "[MB], total = " << total_db << "[MB]." << std::endl;
			std::cout << "__________________________________________________________________________________________________" << std::endl;
		}
		else if (coutMode == 1){
			std::cout << "__________________________________________________________________________________________________" << std::endl;
			std::cout << "    GPU memory usage in stage " << stage << ":" << std::endl;
			std::cout << "                              used  = " << used_db  << "[MB]." << std::endl;
			std::cout << "                              free  = " << free_db  << "[MB]." << std::endl;
			std::cout << "                              total = " << total_db << "[MB]." << std::endl;
			std::cout << "__________________________________________________________________________________________________" << std::endl;
		}
		else{
			std::cerr << "\nERROR: coutMode NO valido." << std::endl;
			exit(-1);
		}
	}
}

void coutDeb(int debug, std::string stringOut){
	if (debug >= 1){
		std::cout << stringOut << std::endl;
	}
}

void cudaSyncGetLastError(cudaError_t error, std::string message){
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess){
		std::cout << "CUDA error " << message << ": " << cudaGetErrorString(error) << std::endl;
		exit(-1);
	}
}

extern "C" void suma_cu(int debug, int coutMode, float *h_resultado, float *h_x, float *h_y, float *h_z, int numFilas){
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