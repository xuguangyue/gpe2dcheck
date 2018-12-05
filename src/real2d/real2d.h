/**
 *    BEC-GP-OMP codes are developed and (c)opyright-ed by:
 *
 *    Luis E. Young-S., Sadhan K. Adhikari
 *    (UNESP - Sao Paulo State University, Brazil)
 *
 *    Paulsamy Muruganandam
 *    (Bharathidasan University, Tamil Nadu, India)
 *
 *    Dusan Vudragovic, Antun Balaz
 *    (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
 *
 *    Public use and modification of this code are allowed provided that the
 *    following three papers are cited:
 *
 *    [1] L. E. Young-S. et al., Comput. Phys. Commun. 204 (2016) 209.
 *    [2] P. Muruganandam, S. K. Adhikari, Comput. Phys. Commun. 180 (2009) 1888.
 *    [3] D. Vudragovic et al., Comput. Phys. Commun. 183 (2012) 2021.
 *
 *    The authors would be grateful for all information and/or comments
 *    regarding the use of the code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <fftw3.h>
#include <complex.h>

#define MAX(a, b) (a > b) ? a : b
#define MAX_FILENAME_SIZE     256
#define RMS_ARRAY_SIZE        5

#define BOHR_RADIUS           5.2917720859e-11
#define FFT_FLAG FFTW_MEASURE

char *output, *initout, *rmsout, *Nstpout, *Npasout, *Nrunout, *dynaout, *tempout, *tempmom;
long outstpx, outstpy, outstpt, outdenstp;

int opt;
long N0;
long Nx, Ny;
long Nx2, Ny2;
long Nstp, Npas, Nrun;
double dx, dy;
double dx2, dy2;
double dt;
double g0_3d, g0_2d, G, Gpar;
double aho, as;
double vgamma, vnu, d_z;
double par;
double pi;
double amp, freq;


double *x, *y;
double *x2, *y2;
double *kx, *ky;
double *kx2, *ky2;
double **pot;
double complex **um1;

fftw_complex *fftin, *fftout;
fftw_plan pf, pb;

void readpar(void);
void initpsi(double complex **, double **);
double randn(double, double);
void calcnorm(double *, double complex **, double **, double **);
void calcmax(double *, double complex **);
void calcmuen(double *, double *, double *, double complex **, double **, double **, double **, double **, double **, double **, double **, double **, double **, double **);
void calcrms(double *, double complex **, double **, double **, double **, double **);
void calcnu(double complex **);
void grad2(double complex **);
void calcfft(double complex **, fftw_complex *, fftw_complex *, fftw_plan);
// void pdcon(double complex **);
void outpsixy(double complex **, FILE *);
// void outpsip(double complex **, FILE *);
void outdenx(double complex **, double *, FILE *);
void outdeny(double complex **, double *, FILE *);

extern double simpint(double, double *, long);
extern void diff(double, double *, double *, long);

extern double *alloc_double_vector(long);
extern double complex *alloc_complex_vector(long);
extern double **alloc_double_matrix(long, long);
extern double complex **alloc_complex_matrix(long, long);
extern void free_double_vector(double *);
extern void free_complex_vector(double complex *);
extern void free_double_matrix(double **);
extern void free_complex_matrix(double complex **);

extern int cfg_init(char *);
extern char *cfg_read(char *);
