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

extern "C" void suma_cu(int debug, int coutMode, float *h_resultado, float *h_x, float *h_y, float *h_z, int numFilas);

void coutDeb(int debug, std::string stringOut);

void getParameters(int argc, char **argv, int *debug, int *coutMode, bool *outputCSV, int *numFilas, std::string *fileOut){

	if (checkCmdLineFlag(argc, (const char **)argv, "debug")){
		debug[0] = getCmdLineArgumentInt(argc, (const char **)argv, "debug");
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
	//else{ //EN CASO DE PARAMETRO OBLIGATORIO
	//	std::cerr << "ERROR: Falta el parametro -fileOut" << std::endl;
	//	exit(EXIT_FAILURE);
	//}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
	//findCudaDevice(argc, (const char **)argv);
	cudaDeviceProp props;
	int devID = 0;
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));

	////////////////////////////////////////////////////////////////////////////////////
	// PARAMETROS DE ENTRADA
	////////////////////////////////////////////////////////////////////////////////////
	int debug           = 0;     // DEFAULT
	int coutMode        = 1;     // DEFAULT
	bool outputCSV      = false; //DEFAULT
	int numFilas        = 5000000; // DEFAULT Numero de filas de entrada
	std::string fileOut = "resultados.csv";  //DEFAULT

	getParameters(argc, argv, &debug, &coutMode, &outputCSV, &numFilas, &fileOut);

	////////////////////////////////////////////////////////////////////////////////////
	// Display INFO
	////////////////////////////////////////////////////////////////////////////////////
	coutDeb(debug, " ============================================================================== ");
	coutDeb(debug, " ------- CUDA TEMPLATE SIMPLE 28/02/2017 -  David Valenzuela Urrutia ---------- ");
	coutDeb(debug, " ============================================================================== ");
	std::string propsMajorStr = std::to_string(props.major);
	std::string propsMinorStr = std::to_string(props.minor);
	std::string computeCuda = "\n        Compute " + propsMajorStr + "." + propsMinorStr + " CUDA device: " + props.name + "\n";
	coutDeb(debug, computeCuda);

	coutDeb(debug, "debug     = " + std::to_string(debug));
	coutDeb(debug, "coutMode  = " + std::to_string(coutMode));
	coutDeb(debug, "outputCSV = " + std::to_string(outputCSV));
	coutDeb(debug, "numFilas  = " + std::to_string(numFilas));
	coutDeb(debug, "fileOut   = " + fileOut);

	// Se crean los arreglos con la data de entrada
	float *h_x;
	h_x = (float*)malloc(numFilas*sizeof(float));

	float *h_y;
	h_y = (float*)malloc(numFilas*sizeof(float));

	float *h_z;
	h_z = (float*)malloc(numFilas*sizeof(float));

	// Se crea un arreglo con datos de RESULTADO
	float *h_resultado;
	h_resultado = (float*)malloc(numFilas*sizeof(float));

	float HI = 10.0;
	float LO =  0.0;
	// Se completan los arreglos con numeros aleatorios
	srand(time(NULL)); // Semilla nueva
	for (int fil = 0; fil < numFilas; fil++){
		h_x[fil] = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));;
		h_y[fil] = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));;
		h_z[fil] = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));;
	}

	////////////////////////////////////////////////////////////////////////////////////
	// Ejecucion codigo .CU
	////////////////////////////////////////////////////////////////////////////////////
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	suma_cu(debug, coutMode, &h_resultado[0], &h_x[0], &h_y[0], &h_z[0], numFilas); // Funcion template.cu
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float timeGPUmillisec = 0;
	cudaEventElapsedTime(&timeGPUmillisec, start, stop);

	////////////////////////////////////////////////////////////////////////////////////
	// WRITE OUTPUT
	////////////////////////////////////////////////////////////////////////////////////
	if (outputCSV == true){
		coutDeb(debug, "WRITING OUTPUT FILE: " + fileOut + "  ...");
		clock_t begin_time;
		begin_time = clock();

		std::ofstream outputFile;
		outputFile.open(fileOut);
		// HEADER CSV
		std::string header = "x,y,z,resultado\n";

		// DATA CSV
		outputFile << header;
		for (int fila = 0; fila < numFilas; fila++){
			outputFile << h_x[fila];
			outputFile << ",";
			outputFile << h_y[fila];
			outputFile << ",";
			outputFile << h_z[fila];
			outputFile << ",";
			outputFile << h_resultado[fila];
			outputFile << "\n";
		}
		outputFile.close();
		float timeCSV = float(clock() - begin_time) / CLOCKS_PER_SEC;
		coutDeb(debug, "Tiempo escritura CSV  = " + std::to_string(timeCSV) + "[s].");
		//coutDeb(debug, "OPENING EXCEL CSV FILE...");
		//system(fileOut.c_str()); //OPENING EXCEL CSV FILE
	}

	cudaDeviceReset(); // Se resetea el dispositivo CUDA
	coutDeb(debug, "##########################################################");
	coutDeb(debug, "Tiempo computo en GPU = " + std::to_string(timeGPUmillisec / (1000.0f)) + "[s].");
	coutDeb(debug, "DONE.");
	return 0;
}