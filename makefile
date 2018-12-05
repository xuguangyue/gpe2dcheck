#
# BEC-GP-OMP codes are developed and (c)opyright-ed by:
#
# Luis E. Young-S., Sadhan K. Adhikari
# (UNESP - Sao Paulo State University, Brazil)
#
# Paulsamy Muruganandam
# (Bharathidasan University, Tamil Nadu, India)
#
# Dusan Vudragovic, Antun Balaz
# (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
#
# Public use and modification of this code are allowed provided that the
# following three papers are cited:
#
# [1] L. E. Young-S. et al., Comput. Phys. Commun. 204 (2016) 209.
# [2] P. Muruganandam, S. K. Adhikari, Comput. Phys. Commun. 180 (2009) 1888.
# [3] D. Vudragovic et al., Comput. Phys. Commun. 183 (2012) 2021.
#
# The authors would be grateful for all information and/or comments
# regarding the use of the code.
#

# CC = icc
# CFLAGS = -fast
# OMPFLAGS = -openmp
# LIBS = -lm
# OMPLIBS = -lpthread

# ifeq ($(compiler), gcc)
	CC = gcc-8 -m64
	CFLAGS = -O3 -Wall
	OMPFLAGS = -fopenmp
	CLIBS = -lfftw3 -lm
	OMPLIBS = -lfftw3_omp
# endif

all: imag2d real2d 

help: ../readme.txt
	less $^

imagsph: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/imagsph/imagsph.c -o imagsph.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o imagsph diffint.o cfg.o mem.o imagsph.o $(CLIBS) $(OMPLIBS)

imagcir: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/imagcir/imagcir.c -o imagcir.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o imagcir diffint.o cfg.o mem.o imagcir.o $(CLIBS) $(OMPLIBS)

imag1d: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/imag1d/imag1d.c -o imag1d.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o imag1d diffint.o cfg.o mem.o imag1d.o $(CLIBS) $(OMPLIBS)

imag2d: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/imag2d/imag2d.c -o imag2d.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o imag2d diffint.o cfg.o mem.o imag2d.o $(CLIBS) $(OMPLIBS)

imagaxi: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/imagaxi/imagaxi.c -o imagaxi.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o imagaxi diffint.o cfg.o mem.o imagaxi.o $(CLIBS) $(OMPLIBS)

imag3d: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/imag3d/imag3d.c -o imag3d.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o imag3d diffint.o cfg.o mem.o imag3d.o $(CLIBS) $(OMPLIBS)

realsph: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/realsph/realsph.c -o realsph.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o realsph diffint.o cfg.o mem.o realsph.o $(CLIBS) $(OMPLIBS)
	rm -rf *.o

realcir: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/realcir/realcir.c -o realcir.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o realcir diffint.o cfg.o mem.o realcir.o $(CLIBS) $(OMPLIBS)
	rm -rf *.o

real1d: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/real1d/real1d.c -o real1d.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o real1d diffint.o cfg.o mem.o real1d.o $(CLIBS) $(OMPLIBS)
	rm -rf *.o

real2d: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/real2d/real2d.c -o real2d.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o real2d diffint.o cfg.o mem.o real2d.o $(CLIBS) $(OMPLIBS)
	rm -rf *.o

realaxi: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/realaxi/realaxi.c -o realaxi.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o realaxi diffint.o cfg.o mem.o realaxi.o $(CLIBS) $(OMPLIBS)
	rm -rf *.o

real3d: diffint cfg mem
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/real3d/real3d.c -o real3d.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o real3d diffint.o cfg.o mem.o real3d.o $(CLIBS) $(OMPLIBS)
	rm -rf *.o

diffint:
	$(CC) $(CFLAGS) -c src/utils/diffint.c -o diffint.o

cfg:
	$(CC) $(CFLAGS) -c src/utils/cfg.c -o cfg.o

mem:
	$(CC) $(CFLAGS) -c src/utils/mem.c -o mem.o

clean:
	rm -rf *~ *.o imag2d real2d
