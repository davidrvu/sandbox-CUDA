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

// Funciones utiles y Clases  
#include "utils.h"
#include "Func1.h"
#include "Func2.h"

void getParameters(int argc, char **argv, int *debug, std::string *mode, int *coutMode, bool *outputCSV, int *numFilas, std::string *fileOut){

	if (checkCmdLineFlag(argc, (const char **)argv, "debug")){
		debug[0] = getCmdLineArgumentInt(argc, (const char **)argv, "debug");
	}

	char *charMode;
	if (getCmdLineArgumentString(argc, (const char **)argv, "mode", &charMode)){
		mode[0].assign(charMode, strlen(charMode));
	}
	else{
		std::cerr << "\nERROR: Falta el parametro -mode" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "coutMode")){
		coutMode[0] = getCmdLineArgumentInt(argc, (const char **)argv, "coutMode");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "outputCSV")){
		outputCSV[0] = getCmdLineArgumentInt(argc, (const char **)argv, "outputCSV");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "numFilas")){
		numFilas[0] = getCmdLineArgumentInt(argc, (const char **)argv, "numFilas");
	}

	char *fnameOut;
	if (getCmdLineArgumentString(argc, (const char **)argv, "fileOut", &fnameOut)){
		fileOut[0].assign(fnameOut, strlen(fnameOut));
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
	clock_t beginTime = clock();

	//findCudaDevice(argc, (const char **)argv);
	cudaDeviceProp props;
	int devID = 0;
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));

	////////////////////////////////////////////////////////////////////////////////////
	// PARAMETROS DE ENTRADA
	////////////////////////////////////////////////////////////////////////////////////
	int debug           = 0;     // DEFAULT
	std::string mode;
	int coutMode        = 1;     // DEFAULT
	bool outputCSV      = false; //DEFAULT
	int numFilas        = 5000000; // DEFAULT Numero de filas de entrada
	std::string fileOut = "resultados.csv";  //DEFAULT

	getParameters(argc, argv, &debug, &mode, &coutMode, &outputCSV, &numFilas, &fileOut);

	////////////////////////////////////////////////////////////////////////////////////
	// Display INFO
	////////////////////////////////////////////////////////////////////////////////////
	coutDeb(debug, " ============================================================================== ");
	coutDeb(debug, " ------- CUDA TEMPLATE CLASS  02/03/2017 -  David Valenzuela Urrutia ---------- ");
	coutDeb(debug, " ============================================================================== ");
	std::string propsMajorStr = std::to_string(props.major);
	std::string propsMinorStr = std::to_string(props.minor);
	std::string computeCuda = "\n        Compute " + propsMajorStr + "." + propsMinorStr + " CUDA device: " + props.name + "\n";
	coutDeb(debug, computeCuda);

	coutDeb(debug, "debug     = " + std::to_string(debug));
	coutDeb(debug, "mode      = " + mode);
	if(mode == "func1"){
		Func1 objetoFunc1(debug, coutMode, outputCSV, numFilas, fileOut);
	}
	else if(mode == "func2"){
		Func2 objetoFunc2(debug, coutMode, outputCSV, numFilas, fileOut);
	}
	else{
		std::cerr << "\nERROR: mode = " << mode << " NO es una entrada valida." << std::endl;
		exit(EXIT_FAILURE);
	}

	clock_t endTime = clock();
	double elapsed_secs = double(endTime - beginTime) / CLOCKS_PER_SEC;

	coutDeb(debug, " ============================================================================== ");
	coutDeb(debug, "mode = " + mode + "done in " + std::to_string(elapsed_secs) + " seconds.");
	return 0;
}