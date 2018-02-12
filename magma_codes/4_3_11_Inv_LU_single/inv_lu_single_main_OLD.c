// David Valenzuela Urrutia 
// GeoInnova
// 8 Marzo 2016
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"

void print_matrix(float *puntero_a_mat, int size){
	int fil, col;
	for(fil = 0; fil < size ; fil++){
		for(col = 0; col < size; col++){
			printf("%f ", puntero_a_mat[(fil*size)+col]);
		}
		printf("\n");
	}
	printf("_______________________\n");
	printf("Mat[m*m-1] = %f \n", puntero_a_mat[size*size-1]);
	printf("Mat[m*m]   = %f \n", puntero_a_mat[size*size]);
	printf("_______________________\n");
}

int main ( int argc , char** argv ){
	magma_init ();   // initialize Magma
	//magma_timestr_t time_start, time_end;
	//float time_start, time_end;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//float gpu_time, *dwork; // dwork - workspace
	float *dwork; // dwork - workspace
	magma_int_t ldwork ;       // size of dwork
	//magma_int_t *piv; // piv - array of indices of inter -
	magma_int_t info;
	magma_int_t *ipiv;
	//magma_int_t m = 8192;      // changed rows ; a - mxm matrix
	//magma_int_t m = 4096;      // changed rows ; a - mxm matrix
	magma_int_t m = 6;      // changed rows ; a - mxm matrix
    magma_int_t mm = m*m ;   // size of a , r , c
	magma_int_t N = m;
    magma_int_t lda    = m;
    magma_int_t n2     = lda*m;
    magma_int_t ldda   = m;  // multiple of 32 by default

    real_Double_t gflops, gpu_perf, gpu_time;


	float *h_A ;              // a - mxm matrix on the host
	float *d_A ;              // d_a - mxm matrix a on the device
	//float *d_r ;              // d_r - mxm matrix r on the device
	//float *d_c ;              // d_c - mxm matrix c on the device
	float *h_Ainv;
	float *h_Ainversa;

	magma_int_t ione = 1;
	magma_int_t ISEED[4] = {0 ,0 ,0 ,1}; // seed
	//magma_int_t errr;

	//const float alpha = 1.0;   // alpha =1
	//const float beta = 0.0;    // beta =0

	ldwork = m * magma_get_sgetri_nb( m ); // workspace size

	gflops = FLOPS_SGETRI(N)/1e9;
	////////////////////////////////////////////////////////////////////////
	// allocate matrices
	////////////////////////////////////////////////////////////////////////
	//errr = magma_smalloc_cpu( &a , mm );     // host memory for a
	//errr = magma_smalloc( &d_a , mm );       // device memory for a
	//errr = magma_smalloc( &d_r , mm );       // device memory for r
	//errr = magma_smalloc( &d_c , mm );       // device memory for c
	//errr = magma_smalloc( &dwork , ldwork ); // dev . mem . for ldwork

	magma_smalloc_cpu( &h_A , mm );     // host memory for a
	magma_smalloc_cpu( &h_Ainv, n2);
	magma_smalloc_cpu( &h_Ainversa, n2);
	//magma_smalloc_cpu( &ipiv,   N);
	magma_smalloc( &d_A , mm );       // device memory for a
	//magma_smalloc( &d_r , mm );       // device memory for r
	//magma_smalloc( &d_c , mm );       // device memory for c
	magma_smalloc( &dwork , ldwork ); // dev . mem . for ldwork

	
	//piv =(magma_int_t*)malloc(m*sizeof( magma_int_t )); // host mem .
	ipiv =(magma_int_t*)malloc(m*sizeof( magma_int_t )); // host mem .
	////////////////////////////////////////////////////////////////////////
	// generate random matrix a                             // for piv
	////////////////////////////////////////////////////////////////////////
	/* Initialize the matrix */
	lapackf77_slarnv(&ione, ISEED, &mm , h_A);            // random h_A

	//todotodo PRINT h_A
	printf("%f\n", h_A[m*m-1]);
	printf("%f\n", h_A[m*m]); // = 0
	printf("%f\n", h_A[m*m+1]); // = 0

	//int size_h_A = (sizeof(h_A))/(sizeof(float*) ); //The compiler doesn't know what the pointer is pointing to
	//printf("size_h_A = %i \n", size_h_A);

	printf("Matriz a invertir = h_A = \n");
	print_matrix(h_A,m);



    // Factor the matrix. Both MAGMA and LAPACK will use this factor.
    magma_ssetmatrix( N, N, h_A, lda, d_A, ldda );  //Copy data from host to device (??)
    magma_sgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
    magma_sgetmatrix( N, N, d_A, ldda, h_Ainv, lda ); //Copy data from device to host (??)
    if (info != 0) {
        printf("magma_sgetrf_gpu returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    }


	printf("h_Ainv = \n");
	print_matrix(h_Ainv,m);

    //===================================================================
    //  Performs operation using MAGMA
    //===================================================================
    gpu_time = magma_wtime();
    magma_sgetri_gpu( N, d_A, ldda, ipiv, dwork, ldwork, &info );
    gpu_time = magma_wtime() - gpu_time;
    gpu_perf = gflops / gpu_time;
    if (info != 0) {
        printf("magma_sgetri_gpu returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    }
  	// Copiar inversa desde GPU
    magma_sgetmatrix( N, N, d_A, ldda, h_Ainversa, lda ); // Inversa FROM device TO host
    printf("h_Ainversa = \n");
	print_matrix(h_Ainversa,m);


	printf( "GPU perf = %7.2f | GPU time = %7.2f  \n", gpu_perf, gpu_time );
    printf("DONE :D \n");



	
	/* OJO REVISAR
	// Factor the matrix. Both MAGMA and LAPACK will use this factor. 

	magma_ssetmatrix(   m, m, a,   m, d_a, m);        // copy a -> d_a
	printf("parte 1 \n");
	magmablas_slacpy( 'A', m, m, d_a,   m, d_r , m );  // copy d_a -> d_r
	printf("parte 2 \n");
	// find the inverse matrix : a_d * X = I using the LU factorization
	// with partial pivoting and row interchanges computed by
	// magma_sgetrf_gpu ; row i is interchanged with row piv ( i );
	// d_a - mxm matrix ; d_a is overwritten by the inverse
	printf("parte 3 \n");

	////time_start = get_current_time();

	cudaEventRecord(start,0); //where 0 is the default stream

	// Factor the matrix. Both MAGMA and LAPACK will use this factor. 
	magma_sgetrf_gpu( m, m, d_a, m, piv, &info);
	// Performs operation using MAGMA
	magma_sgetri_gpu( m,d_a,m,piv,dwork,ldwork,&info);


	cudaEventRecord(stop,0); //where 0 is the default stream
	////time_end = get_current_time();

	//gpu_time = GetTimerValue( time_start , time_end )/1e3;  // Magma time

	magma_sgemm( 'N', 'N', m, m, m, alpha, d_a, m, d_r, m, beta, d_c, m);

	cudaEventSynchronize(stop); //wait for the event to be executed!
	float dt_ms = 0;
	cudaEventElapsedTime(&dt_ms, start, stop);

	//printf ( " magma_sgetrf_gpu + magma_sgetri_gpu time : %7.5f sec . \n" , gpu_time);
	printf ( " magma_sgetrf_gpu + magma_sgetri_gpu time : %7.5f sec . \n" , dt_ms);


	magma_sgetmatrix( m, m, d_c, m, a, m);
	printf ( " upper left corner of a ^ -1* a : \n " );
	magma_sprint( 4 , 4, a , m);  // part of a ^ -1* a
	*/













	
	////////////////////////////////////////////////////////////////////////
	// FREE MEMORY
	////////////////////////////////////////////////////////////////////////	
	//free(a);   // free host memory
	//free(piv); // free host memory
	
	magma_free_cpu(h_A);   // free host memory
	magma_free_cpu(h_Ainv);   // free host memory
	magma_free_cpu(h_Ainversa);   // free host memory
	
	magma_free_cpu(ipiv); // free host memory

	magma_free(d_A); // free device memory
	//magma_free(d_r); // free device memory
	//magma_free(d_c); // free device memory
	magma_finalize();  // finalize Magma
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
