#include <stdio.h>

extern "C" bool funcion_con_cuda(const int argc);

int main(int argc, char **argv){

  printf("Creado por David Valenzuela Urrutia \n");

  bool resultado;
  resultado = funcion_con_cuda(argc);

  printf("%d \n", resultado); // prints 1
  printf(resultado ? "true \n" : "false \n");

}
