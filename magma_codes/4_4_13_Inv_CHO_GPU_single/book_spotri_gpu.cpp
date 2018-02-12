# include <stdio.h>
# include <cuda.h>
# include "magma.h"
# include "magma_lapack.h"

int main (int argc, char ** argv){
	magma_init();                // initialize Magma
	magma_timestr_t start, end;
	float gpu_time;
	magma_int_t info, i, j;
	magma_int_t m = 8192;        // a - mxm matrix
	magma_int_t mm = m*m;    // size of a , r , c
	float *a;       // a - mxm matrix on the host
	float *d_a;    // d_a - mxm matrix a on the device
	float *d_r;    // d_r - mxm matrix r on the device
	float *d_c;    // d_c - mxm matrix c on the device
	magma_int_t ione = 1;
	magma_int_t ISEED[4] = { 0 ,0 ,0 ,1 }; // seed
	magma_err_t err ;
	const float alpha = 1.0; // alpha =1
	const float beta = 0.0;  // beta =0
	// allocate matrices on the host
	err = magma_smalloc_cpu(&a, mm); // host memory for a
	err = magma_smalloc(&d_a, mm);   // device memory for a
	err = magma_smalloc(&d_r, mm);   // device memory for r
	err = magma_smalloc(&d_c, mm);   // device memory for c
	// generate random matrix a
	lapackf77_slarnv(&ione, ISEED, &mm, a); // random a


	// symmetrize a and increase its diagonal. | OJO: IMPORTANTE!!!
	for(i = 0; i<m ; i++) {
		MAGMA_S_SET2REAL(a[i*m + i], (MAGMA_S_REAL(a[i*m + i]) + 1.*m) );
			for(j=0; j<i; j++){
				a[i*m + j] = a[j*m + i];
			}
	}


	magma_ssetmatrix( m,   m, a,   m, d_a,   m);   // copy a -> d_a
	magmablas_slacpy('A', m, m, d_a,   m, d_r, m); // copy d_a -> d_r

	// find the inverse matrix ( d_a )^ -1: d_a * X = I for mxm symmetric
	// positive definite matrix d_a using the Cholesky decomposition obtained by magma_spotrf_gpu

	// d_a is overwritten by the inverse
	start = get_current_time();


	magma_spotrf_gpu(MagmaLower, m, d_a, m, &info);
	magma_spotri_gpu(MagmaLower, m, d_a, m, &info);


	end = get_current_time();
	gpu_time = GetTimerValue(start, end)/1e3; // Magma time


	// compute a ^ -1* a
	magma_ssymm('L', 'L', m, m, alpha, d_a, m, d_r, m, beta, d_c, m);
	printf("magma_spotrf_gpu + magma_spotri_gpu time : %7.5f sec. \n ", gpu_time);
	magma_sgetmatrix( m, m, d_c, m, a, m);   // copy d_c - > a
	printf("upper left corner of a ^ -1* a : \n");
	magma_sprint( 4, 4, a, m);  // part of a ^ -1* a
	free(a);  // free host memory
	magma_free(d_a);  // free device memory
	magma_free(d_r);  // free device memory
	magma_free(d_c);  // free device memory
	magma_finalize(); // finalize Magma
	return 0;
}


// magma_spotrf_gpu + magma_spotri_gpu time : 1.76209 sec .
//
// upper left corner of a ^ -1* a :
// [
//  1.0000   0.0000 -0.0000 0.0000
// -0.0000  1.0000  0.0000 0.0000
//  0.0000  0.0000  1.0000 -0.0000
//  0.0000 -0.0000 -0.0000 1.0000
// ];