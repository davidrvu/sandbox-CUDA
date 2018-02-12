#include <iostream>
#include <armadillo>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>      // printf, scanf, NULL 
#include <stdlib.h>     // malloc, free, rand 
#include <ctime>        // Measure TIME

using namespace std;
using namespace arma;

void print_matrix(double *puntero_a_mat, int size, bool wanna_print_mat){
	if(wanna_print_mat){
		int size_max_print = 5;
		if(size<=size_max_print){
			int fil, col;
			for(fil = 0; fil < size ; fil++){
				for(col = 0; col < size; col++){
					printf("%f ", puntero_a_mat[(fil*size)+col]);
				}
				printf("\n");
			}
			printf("_______________________\n");
		}
		else{
			int fil, col;
			for(fil = 0; fil < size_max_print ; fil++){
				for(col = 0; col < size_max_print; col++){
					printf("%f ", puntero_a_mat[(fil*size)+col]);
				}
				printf("\n");
			}
			printf("_______________________\n");
		}	
	}
}


int main(int argc, char** argv){

	int matrix_size;
	size_t result; 

	if(argc >= 2){
		matrix_size = atoi(argv[1]); // str2num
	}
	else{
		matrix_size = 5;
	}

	bool wanna_print_mat = true;

	// num2str
	int size_str = (int)((ceil(log10(matrix_size))+1)*sizeof(char));
	char matrix_size_str[size_str];
	sprintf(matrix_size_str, "%i", matrix_size);
	printf("matrix_size_str = %s \n", matrix_size_str);

	// FILENAME
	char file_name[80] = "/home/david/matrix_data_base/";
	char file_name_steps[20] =  "mat_";
	strcat(file_name_steps, matrix_size_str);
	strcat(file_name_steps, "x");
	strcat(file_name_steps, matrix_size_str);
	strcat(file_name_steps, "_double_SPD.bin");
	strcat(file_name, file_name_steps);
	printf("file_name = %s \n", file_name);

	FILE *ptr_myfile;
	ptr_myfile=fopen(file_name,"rb");
	if (!ptr_myfile){
		printf("Unable to open file! : %s \n",file_name);
		return 1;
	}

	double *h_mat;
	h_mat = (double*) malloc (sizeof(double)*matrix_size*matrix_size);
  
	double numero;
	int celda;
	for (celda = 0; celda < matrix_size*matrix_size; celda++){
		result = fread(&numero,sizeof(double),1,ptr_myfile);
		// FILE TO VECTOR
		h_mat[celda] = numero;
		//printf("%f \n",numero);
	}
	fclose(ptr_myfile);	

	////////////////////////////////////////////////
	printf(" =========================================================== \n");
	printf("               Load Matrix File \n");
	printf(" =========================================================== \n");
	printf(" Matrix Read from BIN file = \n");
	print_matrix(h_mat,matrix_size,wanna_print_mat);

	printf(" =========================================================== \n");
	printf("               Inv ARMADILLO \n");
	printf(" =========================================================== \n");
	cout << "Armadillo version: " << arma_version::as_string() << endl;

	///////////////////////////////////////////////////////////////////////
	// C++ ARRAY TO ARMADILLO MAT
	///////////////////////////////////////////////////////////////////////
	mat h_mat_ARMA(matrix_size,matrix_size);  // directly specify the matrix size (elements are uninitialised)

	int fil, col;	
	for(fil = 0; fil < matrix_size; fil++){
		for(col = 0; col < matrix_size; col++){
			h_mat_ARMA(fil,col) = (h_mat[fil*matrix_size + col]);
		}
	}
	///////////////////////////////////////////////////////////////////////
	// ARMADILLO MAT PRINT
	///////////////////////////////////////////////////////////////////////
	if(matrix_size <= 5){
		h_mat_ARMA.print("\n h_mat_ARMA = ");
	}
	else{
		mat h_mat_ARMA_submat = h_mat_ARMA.submat(span(0,4), span(0,4));
		h_mat_ARMA_submat.print("\n h_mat_ARMA_submat = ");
	}

	///////////////////////////////////////////////////////////////////////
	// INVERSE
	///////////////////////////////////////////////////////////////////////	
	clock_t start_time = clock();

	mat h_mat_ARMA_inv = inv_sympd(h_mat_ARMA);

	clock_t end_time = clock();

	// print inverse
	if(matrix_size <= 5){
		h_mat_ARMA_inv.print("\n h_mat_ARMA_inv = ");
	}
	else{
		mat h_mat_ARMA_inv_submat = h_mat_ARMA_inv.submat(span(0,4), span(0,4));
		h_mat_ARMA_inv_submat.print("\n h_mat_ARMA_inv_submat = ");
	}

	double total_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
	printf("Elapsed time: %6.10f seconds \n", total_time);

	free (h_mat);
	return 0;
}

