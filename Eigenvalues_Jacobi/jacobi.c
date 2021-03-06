/**
 * File: A baby version of the jacobi method
 * 		Does not sweep
 * 		Finds the largest off diagonal element 
 *		and performs jacobi transformation on the corresponding rows
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void jacobi(double* A, double* V, double c, double s, int p, int q, int n);
void calcJ(double* A, double* c, double* s, int p, int q, int n);

/* calculate the transformaiton matrix J */
void calcJ(double* A, double* cp, double* sp, int p, int q, int n){
	if (*(A + p * n + q) != 0)
	{
		double torque, t;
        torque = ( *(A + q * n + q) - *(A + p * n + p))/(2*(*(A + p * n + q)));
        if (torque >= 0)
            t = 1/(torque + sqrt(1+torque*torque));
        else
            t = -1/(-torque + sqrt(1+torque*torque));
        
        *cp = 1/sqrt(1+t*t);
        *sp = t*(*cp);
    }
    else
    {
        *cp = 1;
        *sp = 0;
	}
}


void jacobi(double* A, double* V, double c, double s, int p, int q, int n){
	int i;
	
	/* A = transpose(J)*A*J */
    for (i = 0; i < n; i++)
    {
        double Api = (*(A + p * n + i))*c + (*(A + q * n + i))*(-s);
        double Aqi = (*(A + p * n + i))*s + (*(A + q * n + i))*c;
        *(A + p * n + i) = Api;
        *(A + q * n + i) = Aqi;
    }

    
    for (i = 0; i < n; i++)
    { 
        double Aip = (*(A + i * n + p))*c + (*(A + i * n + q))*(-s);
        double Aiq = (*(A + i * n + p))*s + (*(A + i * n + q))*c;
        *(A + i * n + p) = Aip;
        *(A + i * n + q) = Aiq;
    }
     
    /* V = V*J */
    for (i = 0; i < n; i++)
    { 
        double Vpi = (*(V + p * n + i))*c + (*(V + q * n + i))*(-s);
        double Vqi = (*(V + p * n + i))*s + (*(V + q * n + i))*c;
        *(V + p * n + i) = Vpi;
        *(V + q * n + i) = Vqi;
	}
}


int main (void){

	double eps = 0.000000000001;
	int n, row, col, p, q;
	double max_off = 0;
	double c, s;
	double *cp, *sp;
	cp = &c;
	sp = &s;

	double* A = (double*)malloc(100*100*sizeof(double));
	double* V = (double*)malloc(100*100*sizeof(double));

	/* enter a valid matrix A*/
	*A = 1.0;
	*(A+1) = 1.0;
	*(A+2) = 1.0;
	*(A+3) = 1.0;
	*(A+4) = 1.0;
	*(A+5) = 2.0;
	*(A+6) = 3.0;
	*(A+7) = 4.0;
	*(A+8) = 1.0;
	*(A+9) = 3.0;
	*(A+10) =6.0;
	*(A+11) = 10.0;
	*(A+12) = 1.0;
	*(A+13) = 4.0;
	*(A+14) = 10.0;
	*(A+15) = 20.0;
	n = 4;

	/*initializing vector matrix V */
	for (row = 0; row < n; row++) 
	{
		for (col = 0; col < n; col++) 
		{
			if (row == col)
				*(V + row * n + col) = 1.0;
			else
				*(V + row * n + col) = 0.0;
		}
	}

	/*find largest off diagonal*/
	for (row = 0; row < n; row++) {
		for (col = 0; col < n; col++) {
			if (row != col)
			{
				if (fabsf(*(A + row * n + col)) > max_off)
				{
					max_off = fabsf(*(A + row * n + col));
					if (row < col)
					{
						p = row;
						q = col;
					}
					else 
					{
						p = col;
						q = row;
					}
				}
			}
		}
	}
	printf("max_off = %f\n", max_off);

	while (fabs(max_off) > eps)
	{
		calcJ(A, cp, sp, p, q, n);
		jacobi(A, V, c, s, p, q, n);

		double* pt = A;
		for (; pt < A + n*n; pt++)
			printf("%f\n", *pt);
		printf("\n");

		max_off = 0;
		for (row = 0; row < n; row++) {
			for (col = 0; col < n; col++) {
				if (row != col)
				{
					if (fabsf(*(A + row * n + col)) > max_off)
					{
						max_off = fabsf(*(A + row * n + col));
						if (row < col)
						{
							p = row;
							q = col;
						}
						else 
						{
							p = col;
							q = row;
						}
					}
				}
			}
		}
		
		printf("max_off = %f \n", max_off);
		printf("\n");	
	}

	double* pt = A;
	for (; pt < A + n*n; pt++)
		printf("%f\n", *pt);
	printf("\n");
	printf("max_off = %f", max_off);
	printf("\n");
}