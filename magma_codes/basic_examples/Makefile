# This is a simple standalone example. See README.txt
# Initially it is setup to use OpenBLAS.
# See magma/make.inc for alternate BLAS and LAPACK libraries,

# David Valenzuela Urrutia 26 Febrero 2016

# Paths where MAGMA, CUDA, and OpenBLAS are installed

MAGMADIR     = /usr/local/magma
CUDADIR      = /usr/local/cuda-7.5
OPENBLASDIR  = /usr/local/openblas
LAPACKDIR    = /usr/lib/lapack         

CC            = gcc
FORT          = gfortran
LD            = gcc
CFLAGS        = -Wall
LDFLAGS       = -Wall

MAGMA_CFLAGS   := -DADD_ -I$(MAGMADIR)/include -I$(CUDADIR)/include
MAGMA_F90FLAGS := -I$(MAGMADIR)/include -Dmagma_devptr_t="integer(kind=8)"

#MAGMA_LIBS   := -L$(MAGMADIR)/lib -L$(CUDADIR)/lib64 -L$(OPENBLASDIR)/lib \
                -lmagma -lcublas -lcudart -lopenblas -lpthread -lstdc++ -lm -lcuda

MAGMA_LIBS   := -L$(MAGMADIR)/lib -L$(CUDADIR)/lib64 -L$(OPENBLASDIR)/lib -L$(LAPACKDIR) \
                -lmagma -lcublas -lcudart -lopenblas -lpthread -lstdc++ -lm -lcuda -llapack -lcusparse -fopenmp

# ----------------------------------------
default: all

all: example_v1 example_v2

clean:
	-rm -f example_v1 example_v2 *.o *.mod

.SUFFIXES:
# ----------------------------------------
%.o: %.c
	$(CC) $(CFLAGS) $(MAGMA_CFLAGS) -c -o $@ $<

example_v1: example_v1.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)

example_v2: example_v2.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)