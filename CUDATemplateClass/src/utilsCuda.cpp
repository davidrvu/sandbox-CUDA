#include "utilsCuda.h"

void cudaSyncGetLastError(cudaError_t error, std::string message){
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess){
		std::cout << "CUDA error " << message << ": " << cudaGetErrorString(error) << std::endl;
		exit(-1);
	}
}

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