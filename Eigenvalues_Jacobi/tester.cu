/* compile with nvcc -arch=sm_35 -o tester jacobi_f.o jacobi_cpu.o */

// OJO COMPILAR: nvcc -arch=sm_35 tester.cu -o tester jacobi_f.o jacobi_cpu.o
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "jacobi_f.h"
#include "jacobi_cpu.h"

void matrix_print( double* M, int n){
	int row, col;
	printf("M = \n");	
	for (row = 0; row < n; row++) {
		for (col = 0; col < n; col++) {
			printf("%10.3f ",*(M + (row * n) + col) );
			//printf("A ");
		}
		printf("\n");
	}
	printf("\n");
}

/*
c =
   1   2   3
   2   4   5
   3   5   6

>> eig(c)
ans =
   -0.51573
    0.17092
   11.34481
*/

int main(int argc, char *argv[]){

	if(argc != 2){
		printf("not enough arguments. supply matrix size. \n ");
		exit(1);
	}

	int size = atoi(argv[1]);
	
	double* M = (double*)malloc(size*size*sizeof(double));
	double valor_celda;
	int row, col;
	clock_t start, diff;

	// TRIANGULAR INFERIOR random
	// DAVID TEST (Borrar)
	if(size == 3){
		*(M + 0) = 1;
		*(M + 3) = 2;
		*(M + 4) = 4;
		*(M + 6) = 3;
		*(M + 7) = 5;
		*(M + 8) = 6;
	}
	else{
		for (row = 0; row < size; row++){
			for (col = 0; col <= row; col++){
				valor_celda = (double)rand()/(double)RAND_MAX*100;
				//printf("%10.3f \n", valor_celda);
				*(M + (row*size) +col) = valor_celda;		
			}
		}		
	}

	matrix_print(M, size);
	
	// TRIANGULAR SUPERIOR (copia) para que quede SIMETRICA 
	for (row = 0; row < size; row++){
		//for (col = 0; col < (size - row - 1); col++){
		for (col = 0; col <= row; col++){	
			//printf("row = %i | col = %i \n",row,col);
			*(M + (col*size) +row) = *(M+ (row*size) +col);	
		}	
	}

	printf("For matrix size %d*%d\n", size, size);

	matrix_print(M, size);

	/* recoding time for serial version*/
	start = clock();
	jacobi_c(M,size);  // ARCHIVO: jacobi_cpu.cu
	diff = clock() - start;

	double t_in_sec = (double)diff/(double)CLOCKS_PER_SEC;
	printf("Time taken for CPU jacobi: %f seconds.\n", t_in_sec);
	
	/* recording time for parallel version */
	start = clock();
	jacobi_cu(M,size);  // ARCHIVO: jacobi_f.cu
	diff = clock() - start;

	t_in_sec = (double)diff/(double)CLOCKS_PER_SEC;
	printf("Time taken for GPU jacobi: %f seconds.\n", t_in_sec);

	return 0;

}

