PASO 1 COMPILAR (directamente):
$nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

PASO 2 EJECUTAR:
$./hello-world

___________________________________________________________________________________

PASO 0 BORRAR COMPILACIÓN ANTERIOR:
$make clean 

PASO 1 COMPILAR (usando archivo Makefile):
$make

PASO 2 EJECUTAR:
$make run
