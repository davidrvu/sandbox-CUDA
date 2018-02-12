// David Valenzuela Urrutia 
// GeoInnova
// 10 Marzo 2016
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

void print_matrix(float *puntero_a_mat, int size, bool wanna_print_mat){
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

	float *dwork; // dwork - workspace
	real_Double_t gflops, gpu_perf;
	real_Double_t gpu_time_inicial, gpu_time_final, gpu_time_delta;
	magma_int_t ldwork ;       // size of dwork
	magma_int_t info;
	//magma_int_t *ipiv;

	magma_int_t m = matrix_size;      // mxm matrix

	bool wanna_print_mat = true;
    magma_int_t mm = m*m ;  // size of a , r , c
    
	float *h_A ;      // h_A - mxm matrix on the host
	float *d_A ;      // d_A - mxm matrix a on the device
	//float *h_Ainv;
	float *h_Ainversa;

	magma_int_t ione = 1;
	magma_int_t ISEED[4] = {0 ,0 ,0 ,1}; // seed
	ldwork = m * magma_get_sgetri_nb( m ); // workspace size
	gflops = FLOPS_SGETRI(m)/1e9;

	//////////////////////////////////////////////////////
	//////////////////////////////////////////////////////
	//magma_opts opts;
    //opts.parse_opts( argc, argv );
    //opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    //float tol = opts.tolerance * lapackf77_slamch("E");
    //printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////

	printf(" =========================================================== \n");
	printf("               4_3_13_Inv_CHO_GPU_single \n");
	printf(" =========================================================== \n");
	printf(" Matriz size = %i \n", m);
	printf(" ___________________________________________________________ \n");

	//===================================================================
	// Allocate matrices
	//===================================================================
	printf(" Allocate matrices ... \n");
	magma_smalloc_cpu(&h_A , mm );     // host memory for a
	//magma_smalloc_cpu(&h_Ainv, mm);
	magma_smalloc_cpu(&h_Ainversa, mm);
	magma_smalloc(&d_A , mm );       // device memory for a
	magma_smalloc(&dwork , ldwork ); // dev . mem . for ldwork

	//magma_smalloc_cpu(&ipiv,m); // <- NO FUNCIONA (?)
	//ipiv = (magma_int_t*)malloc(m*sizeof( magma_int_t )); // host mem 


	//===================================================================
	// Generate random matrix h_A                             // for piv
	//===================================================================
	// Initialize the matrix 
	printf(" Generate random matrix h_A  ... \n \n");
	lapackf77_slarnv(&ione, ISEED, &mm , h_A);            // random h_A

	// Symmetrize h_A and increase its diagonal. | OJO: IMPORTANTE!!! magma_smake_hpd(m, h_A, m);
	// Make a matrix symmetric/symmetric positive definite. Increases diagonal by N, and makes it real.
    // Sets Aji = conj( Aij ) for j < i, that is, copy lower triangle to upper triangle.
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

	//===================================================================
	// Factor the matrix. Both MAGMA and LAPACK will use this factor.
	//===================================================================
	printf(" Factor the matrix. Both MAGMA and LAPACK will use this factor  ... \n");
    
	gpu_time_inicial = magma_wtime();

    magma_ssetmatrix( m, m, h_A, m, d_A, m );      //Copy data FROM host TO device
    magma_uplo_t uplo = {MagmaLower};
    printf(" uplo = %s     ===== OK \n", lapack_uplo_const(uplo) ); //opts.uplo = 'Lower';
    magma_spotrf_gpu(uplo, m, d_A, m, &info);
    //magma_sgetrf_gpu( m, m, d_A, m, ipiv, &info ); // de LU float code

    //===================================================================
    //  Operation using MAGMA
    //===================================================================

    magma_spotri_gpu(uplo, m, d_A, m, &info );
    //magma_sgetri_gpu( m, d_A, m, ipiv, dwork, ldwork, &info ); //de LU float code

  	// Copiar inversa desde GPU (device) a CPU (host)
    magma_sgetmatrix( m, m, d_A, m, h_Ainversa, m ); // Inversa FROM device TO host
    gpu_time_final = magma_wtime();

    if (info != 0){
    	printf("magma_spotri_gpu returned error %d: %s.\n", (int)info, magma_strerror(info));
	}
	//Copia triangular superior de la invertida, a su triangular inferior.
    for( i=0; i < m; ++i ){
        for( j=0; j < i; ++j ) {
            h_Ainversa[j + i*m] = h_Ainversa[i + j*m];
        }
    }	

    printf(" h_Ainversa (TAMBIÉN ES SIMÉTRICA) = \n");
	print_matrix(h_Ainversa,m,wanna_print_mat);

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
	magma_free_cpu(h_A);          // free host memory
	magma_free_cpu(h_Ainversa);   // free host memory
	magma_free(d_A);              // free device memory

	magma_finalize();  // finalize Magma
	printf(" DONE !! \n");
	return 0;
}

// EJEMPLO RESULTADO
// magma_sgetrf_gpu + magma_sgetri_gpu time : 2.13416 sec .
//
// upper left corner of a ^ -1* a :
// [
//  1.0000 0.0000 -0.0000 -0.0000
//  0.0000 1.0000 -0.0000 -0.0000
//  0.0000 0.0000  1.0000  0.0000
// -0.0000 -0.0000 0.0000  1.0000
// ];
