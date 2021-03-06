// David Valenzuela Urrutia 
// GeoInnova
// 14 Marzo 2016
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "magma_types.h"

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
		//printf("Mat[m*m-1] = %f \n", puntero_a_mat[size*size-1]);
		//printf("Mat[m*m]   = %f \n", puntero_a_mat[size*size]);
		//printf("_______________________\n");		
	}
}

int main ( int argc , char** argv ){
	magma_init ();   // initialize Magma

	int matrix_size;
	if(argc >= 2){
		matrix_size = atoi(argv[1]); // str2num
	}
	else{
		matrix_size = 5;
	}

	FILE *ptr_myfile;
	bool wanna_print_mat = true;

	real_Double_t gflops, gpu_perf;
	real_Double_t gpu_time_inicial, gpu_time_final, gpu_time_delta;
	magma_int_t info;

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

	magma_int_t m = matrix_size;      // mxm matrix
    magma_int_t mm = m*m ;  // size of a , r , c
    
    double *h_mat_in;
	gflops = FLOPS_SGETRI(m)/1e9;

	//===================================================================
	// Load Matrix File
	//===================================================================
	
	ptr_myfile=fopen(file_name,"rb");
	if (!ptr_myfile){
		printf("Unable to open file! : %s \n",file_name);
		return 1;
	}

	magma_dmalloc_cpu(&h_mat_in, mm);   
	double numero;
	int celda;
	for (celda = 0; celda < matrix_size*matrix_size; celda++){
		fread(&numero,sizeof(double),1,ptr_myfile);
		h_mat_in[celda] = numero;
		//printf("%f \n",numero);
	}
	fclose(ptr_myfile);

	printf(" =========================================================== \n");
	printf("               Load Matrix File \n");
	printf(" =========================================================== \n");
	printf(" Matrix Read from BIN file (SIMETRICA Y DEFINIDA POSITIVA) = \n");
	print_matrix(h_mat_in,matrix_size,wanna_print_mat);

	printf(" =========================================================== \n");
	printf("               4_4_12_Inv_CHO_CPU_double \n");
	printf(" =========================================================== \n");
	printf(" Matriz size = %i \n", m);
	printf(" ___________________________________________________________ \n");

	//===================================================================
	// Allocate matrices
	//===================================================================
	printf(" Allocate matrices ... \n");

	//===================================================================
	// Generate random matrix h_A                             // for piv
	//===================================================================
	// Initialize the matrix 
	//printf(" Generate random matrix h_A  ... \n \n");
	//lapackf77_slarnv(&ione, ISEED, &mm , h_A);            // random h_A

	// Symmetrize h_A and increase its diagonal. | OJO: IMPORTANTE!!! magma_smake_hpd(m, h_A, m);
	// Make a matrix symmetric/symmetric positive definite. Increases diagonal by N, and makes it real.
    // Sets Aji = conj( Aij ) for j < i, that is, copy lower triangle to upper triangle.
	/*
	magma_int_t i, j;
    for( i=0; i < m; ++i ){
   		// #define A(i,j)  A[i + j*lda]        <- OJO!! MUY UTIL!!
        h_A[i + i*m] = MAGMA_S_MAKE( MAGMA_S_REAL( h_A[i + i*m] ) + m, 0. );
        //h_A[i + i*m] = MAGMA_S_MAKE( MAGMA_S_REAL( h_A[i + i*m] ) + 1, 0. );
        for( j=0; j < i; ++j ) {
            h_A[j + i*m] = MAGMA_S_CONJ( h_A[i + j*m] );
        }
    }
	printf(" Matriz a invertir (SIMETRICA Y DEFINIDA POSITIVA) = h_A = \n");
	print_matrix(h_A,m,wanna_print_mat);
	*/
	//===================================================================
	// Factor the matrix. Both MAGMA and LAPACK will use this factor.
	//===================================================================
	printf(" Factor the matrix. Both MAGMA and LAPACK will use this factor  ... \n");
    
	gpu_time_inicial = magma_wtime();

    magma_uplo_t uplo = {MagmaLower};
    printf(" uplo = %s     ===== OK \n", lapack_uplo_const(uplo) ); //opts.uplo = 'Lower';
    
    magma_dpotrf(uplo, m, h_mat_in, m, &info);

    //===================================================================
    //  Operation using MAGMA
    //===================================================================

	magma_dpotri(uplo, m, h_mat_in, m, &info );

    gpu_time_final = magma_wtime();

    if (info != 0){
    	printf("magma_dpotri returned error %d: %s.\n", (int)info, magma_strerror(info));
	}
	//Copia triangular superior de la invertida, a su triangular inferior.
    magma_int_t i, j;
    for( i=0; i < m; ++i ){
        for( j=0; j < i; ++j ) {
            h_mat_in[j + i*m] = h_mat_in[i + j*m];
        }
    }	

    printf(" h_Ainversa (TAMBIÉN ES SIMÉTRICA) = \n");
	print_matrix(h_mat_in,m,wanna_print_mat);

	//===================================================================
	// GPU performance
	//===================================================================
    gpu_time_delta = gpu_time_final - gpu_time_inicial;
    gpu_perf = gflops / gpu_time_delta;
	printf(" GPU perf = %7.2f | GPU time = %7.10f  \n", gpu_perf, gpu_time_delta );
	
	//===================================================================
	// Free Memory
	//===================================================================
	printf(" Free Memory ... \n");	
	magma_free_cpu(h_mat_in);

	magma_finalize();  // finalize Magma
	printf(" DONE !! \n");
	return 0;
}