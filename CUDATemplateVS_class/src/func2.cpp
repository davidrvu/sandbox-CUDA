// Funciones utiles y Clases  
#include "utils.h"
#include "Func2.h"

Func2::Func2(
	int debug,
	int coutMode,
	int outputCSV,
	int numFilas,
	std::string fileOut){

	coutDeb(debug, " Constructor de la clase Func2");

	func2Main(debug, coutMode, outputCSV, numFilas, fileOut);

}

int Func2::func2Main(int debug, int coutMode, int outputCSV, int numFilas, std::string fileOut){
	coutDeb(debug, "----> Func2::func2Main");

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
	float LO = 0.0;
	// Se completan los arreglos con numeros aleatorios
	srand(static_cast<unsigned int>(time(NULL))); // Semilla nueva
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
	mult_cu(debug, coutMode, &h_resultado[0], &h_x[0], &h_y[0], &h_z[0], numFilas); // Funcion template.cu
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float timeGPUmillisec = 0;
	cudaEventElapsedTime(&timeGPUmillisec, start, stop);

	////////////////////////////////////////////////////////////////////////////////////
	// WRITE OUTPUT
	////////////////////////////////////////////////////////////////////////////////////
	if (outputCSV == 1){ // 1 = true
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