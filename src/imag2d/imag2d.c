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
 *
 *    This program solves the time-independent Gross–Pitaevskii nonlinear
 *    partial differential equation in two space dimensions in anisotropic
 *    trap using the imaginary-time propagation. The Gross–Pitaevskii equation
 *    describes the properties of a dilute trapped Bose–Einstein condensate.
 *    The equation is solved using the split-step Crank–Nicolson method by
 *    discretizing space and time. The discretized equation is then propagated
 *    in imaginary time over small time steps. When convergence is achieved,
 *    the method has yielded the stationary solution of the problem.
 *
 *    Description of variables used in the code:
 *
 *    opt     - decides which rescaling of GP equation will be used
 *    par     - parameter for rescaling of GP equation
 *    psi     - array with the wave function values
 *    pot     - array with the values of the potential
 *    g_2d    - final nonlinearity
 *    norm    - wave function norm
 *    rms     - root mean square radius
 *    mu      - chemical potential
 *    en      - energy
 *    Nx      - number of discretization points in the x-direction
 *    Ny      - number of discretization points in the y-direction
 *    x       - array with the space mesh values in the x-direction
 *    y       - array with the space mesh values in the y-direction
 *    dx      - spatial discretization step in the x-direction
 *    dy      - spatial discretization step in the y-direction
 *    dt      - time discretization step
 *    vgamma  - Gamma coefficient of anisotropy of the trap (omega_x / omega)
 *    vnu     - Nu coefficient of anisotropy of the trap (omega_y / omega)
 *    Nstp    - number of initial iterations to introduce the nonlinearity g_2d
 *    Npas    - number of subsequent iterations with the fixed nonlinearity g_2d
 *    Nrun    - number of final iterations with the fixed nonlinearity g_2d
 *    output  - output file with the summary of final values of all physical
 *              quantities
 *    initout - output file with the initial wave function
 *    Npasout - output file with the wave function obtained after the
 *              subsequent Npas iterations, with the fixed nonlinearity g_2d
 *    Nrunout - output file with the final wave function obtained after the
 *              final Nrun iterations
 *    outstpx - discretization step in the x-direction used to save wave
 *              functions
 *    outstpy - discretization step in the yG-direction used to save wave
 *              functions
 */

#include "imag2d.h"

int main(int argc, char **argv) {
   FILE *out;
   FILE *file;
   FILE *filerms;
   char filename[MAX_FILENAME_SIZE];
   int nthreads;
   long cnti;
   double norm, mu, en;
   double *rms;
   double **psi;
   double **cbeta;
   double **dpsix, **dpsiy;
   double **tmpxi, **tmpyi, **tmpxj, **tmpyj;
   double *tmpx, *tmpy;
   double psi2;

   time_t clock_beg, clock_end;
   clock_beg = time(NULL);

   pi = 4. * atan(1.);

   if((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) {
      fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
      exit(EXIT_FAILURE);
   }

   if(! cfg_init(argv[2])) {
      fprintf(stderr, "Wrong input parameter file.\n");
      exit(EXIT_FAILURE);
   }

   readpar();

   #pragma omp parallel
      #pragma omp master
         nthreads = omp_get_num_threads();

   rms = alloc_double_vector(RMS_ARRAY_SIZE);
   x = alloc_double_vector(Nx);
   y = alloc_double_vector(Ny);

   x2 = alloc_double_vector(Nx);
   y2 = alloc_double_vector(Ny);

   pot = alloc_double_matrix(Nx, Ny);
   psi = alloc_double_matrix(Nx, Ny);

   dpsix = alloc_double_matrix(Nx, Ny);
   dpsiy = alloc_double_matrix(Nx, Ny);

   calphax = alloc_double_vector(Nx - 1);
   calphay = alloc_double_vector(Ny - 1);
   cbeta =  alloc_double_matrix(nthreads, MAX(Nx, Ny) - 1);
   cgammax = alloc_double_vector(Nx - 1);
   cgammay = alloc_double_vector(Ny - 1);

   tmpxi = alloc_double_matrix(nthreads, Nx);
   tmpyi = alloc_double_matrix(nthreads, Ny);
   tmpxj = alloc_double_matrix(nthreads, Nx);
   tmpyj = alloc_double_matrix(nthreads, Ny);

   tmpx = alloc_double_vector(Nx);
   tmpy = alloc_double_vector(Ny);

   if(output != NULL) {
      sprintf(filename, "%s.txt", output);
      out = fopen(filename, "w");
   }
   else out = stdout;

   if(rmsout != NULL) {
      sprintf(filename, "%s.txt", rmsout);
      filerms = fopen(filename, "w");
   }
   else filerms = NULL;

   fprintf(out, " Imaginary time propagation 2d,   OPTION = %d\n\n", opt);
   fprintf(out, "  Number of Atoms N = %li, Unit of length AHO = %.8f m\n", Na, aho);
   fprintf(out, "  Scattering length a = %.2f*a0\n", as);
   fprintf(out, "  Nonlinearity G_3D = %.7f\n", g_3d);
   fprintf(out, "  Nonlinearity G_2D = %.7f\n", g_2d);
   fprintf(out, "  Parameters of trap: GAMMA = %.2f, NU = %.2f\n", vgamma, vnu);
   fprintf(out, "  Axial trap parameter = %.2f\n\n", d_z);
   fprintf(out, " # Space Stp: NX = %li, NY = %li\n", Nx,Ny);
   fprintf(out, "  Space Step: DX = %.4f, DY = %.4f\n", dx,dy);
   fprintf(out, " # Time Stp : NSTP = %li, NPAS = %li, NRUN = %li\n", Nstp, Npas, Nrun);
   fprintf(out, "   Time Step:   DT = %.6f\n\n",  dt);
   fprintf(out, "                  --------------------------------------------------------\n");
   fprintf(out, "                    Norm      Chem        Ener/N     <rho>     |Psi(0,0)|^2\n");
   fprintf(out, "                  --------------------------------------------------------\n");
   fflush(out);

   printf("initialization!\n");
   init(psi);
   gencoef();
   calcnorm(&norm, psi, tmpxi, tmpyi);
   calcmuen(&mu, &en, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj);
   calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
   calcmax(&psi2,psi);
   fprintf(out, "Initial : %15.4f %11.6f %11.6f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
   fflush(out);

   if(initout != NULL) {
      sprintf(filename, "%s.txt", initout);
      file = fopen(filename, "w");
      outpsi2xy(psi, file);
      fclose(file);

      sprintf(filename, "%s1d_x.txt", initout);
      file = fopen(filename, "w");
      outdenx(psi, tmpy, file);
      fclose(file);

      sprintf(filename, "%s1d_y.txt", initout);
      file = fopen(filename, "w");
      outdeny(psi, tmpx, file);
      fclose(file);
   }

   if(rmsout != NULL) {
      fprintf(filerms, " Imaginary time propagation 2d,   OPTION = %d\n\n", opt);
      fprintf(filerms, "                  ------------------------------------------------------------\n");
      fprintf(filerms, "Values of rms size:    <x>         <x2>        <y>         <y2>        <rho>\n");
      fprintf(filerms, "                  ------------------------------------------------------------\n");
      fprintf(filerms, "           Initial:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
      fflush(filerms);
   }

   if(Nstp != 0) {
      printf("%ld step evolution to ground state\n", Nstp);
      double g_stp = par * g_2d / (double) Nstp;
      G = 0.;
      for(cnti = 0; cnti < Nstp; cnti ++) {
         G += g_stp;
         calcnu(psi);
         calclux(psi, cbeta);
         calcluy(psi, cbeta);
         calcnorm(&norm, psi, tmpxi, tmpyi);
         printf("%ld\n", cnti);
      }
      calcmuen(&mu, &en, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj);
      calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
      calcmax(&psi2,psi);
      fprintf(out, "After NSTP iter.:%8.4f %11.6f %11.6f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
      fflush(out);

      if(Nstpout != NULL) {
         sprintf(filename, "%s.txt", Nstpout);
         file = fopen(filename, "w");
         outpsi2xy(psi, file);
         fclose(file);

         sprintf(filename, "%s1d_x.txt", Nstpout);
         file = fopen(filename, "w");
         outdenx(psi, tmpy, file);
         fclose(file);

         sprintf(filename, "%s1d_y.txt", Nstpout);
         file = fopen(filename, "w");
         outdeny(psi, tmpx, file);
         fclose(file);
      }

      if(rmsout != NULL) {
         fprintf(filerms, "  After NSTP iter.:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
         fflush(filerms);
      }
   }
   else {
      G = par * g_2d;
   }

   printf("%ld step imaginary evolution\n", Npas);
   for(cnti = 0; cnti <= Npas; cnti ++) {
      calcnu(psi);
      calclux(psi, cbeta);
      calcluy(psi, cbeta);
      calcnorm(&norm, psi, tmpxi, tmpyi);
      printf("%ld %9.5f\n", cnti, norm);
   }
   if(Npas != 0){
      calcmuen(&mu, &en, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj);
      calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
      calcmax(&psi2,psi);
      fprintf(out, "After NPAS iter.:%8.4f %11.6f %11.6f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
      fflush(out);

      if(rmsout != NULL) {
         fprintf(filerms, "  After NPAS iter.:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
         fflush(filerms);
      }

      if(Npasout != NULL) {
         sprintf(filename, "%s.txt", Npasout);
         file = fopen(filename, "w");
         outpsi2xy(psi, file);
         fclose(file);

         sprintf(filename, "%s1d_x.txt", Npasout);
         file = fopen(filename, "w");
         outdenx(psi, tmpy, file);
         fclose(file);

         sprintf(filename, "%s1d_y.txt", Npasout);
         file = fopen(filename, "w");
         outdeny(psi, tmpx, file);
         fclose(file);
      }
   }

   printf("%ld step check evolution\n", Nrun);
   for(cnti = 0; cnti <= Nrun; cnti ++) {
      calcnu(psi);
      calclux(psi, cbeta);
      calcluy(psi, cbeta);
      calcnorm(&norm, psi, tmpxi, tmpyi);
      printf("%ld %9.5f\n", cnti, norm);
   }
   if(Nrun != 0){
      calcmuen(&mu, &en, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj);
      calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
      calcmax(&psi2,psi);
      fprintf(out, "After NRUN iter.:%8.4f %11.6f %11.6f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
      fflush(out);

      if(rmsout != NULL) {
         fprintf(filerms, "  After NRUN iter.:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
         fprintf(filerms, "                  ------------------------------------------------------------\n");
      }

      if(Nrunout != NULL) {
         sprintf(filename, "%s.txt", Nrunout);
         file = fopen(filename, "w");
         outpsi2xy(psi, file);
         fclose(file);

         sprintf(filename, "%s1d_x.txt", Nrunout);
         file = fopen(filename, "w");
         outdenx(psi, tmpy, file);
         fclose(file);

         sprintf(filename, "%s1d_y.txt", Nrunout);
         file = fopen(filename, "w");
         outdeny(psi, tmpx, file);
         fclose(file);
      }
   }

   if(rmsout != NULL) fclose(filerms);

   fprintf(out, "                  --------------------------------------------------------\n\n");

   free_double_vector(rms);

   free_double_vector(x);
   free_double_vector(y);

   free_double_vector(x2);
   free_double_vector(y2);

   free_double_matrix(pot);
   free_double_matrix(psi);

   free_double_matrix(dpsix);
   free_double_matrix(dpsiy);

   free_double_vector(calphax);
   free_double_vector(calphay);
   free_double_matrix(cbeta);
   free_double_vector(cgammax);
   free_double_vector(cgammay);

   free_double_matrix(tmpxi);
   free_double_matrix(tmpyi);
   free_double_matrix(tmpxj);
   free_double_matrix(tmpyj);

   free_double_vector(tmpx);
   free_double_vector(tmpy);

   clock_end = time(NULL);
   double wall_time = difftime(clock_end, clock_beg);
   double cpu_time = clock() / (double) CLOCKS_PER_SEC;
   fprintf(out, " Clock Time: %.f seconds\n", wall_time);
   fprintf(out, " CPU Time: %.f seconds\n", cpu_time);

   if(output != NULL) fclose(out);

   return(EXIT_SUCCESS);
}

/**
 *    Reading input parameters from the configuration file.
 */
void readpar(void) {
   char *cfg_tmp;

   if((cfg_tmp = cfg_read("OPTION")) == NULL) {
      fprintf(stderr, "OPTION is not defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   opt = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("D_Z")) == NULL) {
      fprintf(stderr, "D_Z is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   d_z = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("G_2D")) == NULL) {
      if((cfg_tmp = cfg_read("NATOMS")) == NULL) {
      	fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
      	exit(EXIT_FAILURE);
      }
      Na = atol(cfg_tmp);

      if((cfg_tmp = cfg_read("AHO")) == NULL) {
         fprintf(stderr, "AHO is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      aho = atof(cfg_tmp);

      if((cfg_tmp = cfg_read("AS")) == NULL) {
         fprintf(stderr, "AS is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      as = atof(cfg_tmp);

      g_3d = 4. * (pi) * as * Na * BOHR_RADIUS / aho;
      g_2d = g_3d / (sqrt(2. * (pi)) * d_z);
   } else {
      g_2d = atof(cfg_tmp);
      g_3d = g_2d * (sqrt(2. * (pi)) * d_z);
   }

   if((cfg_tmp = cfg_read("NX")) == NULL) {
      fprintf(stderr, "NX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nx = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("NY")) == NULL) {
      fprintf(stderr, "NY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Ny = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("DX")) == NULL) {
      fprintf(stderr, "DX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dx = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("DY")) == NULL) {
      fprintf(stderr, "DY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dy = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("DT")) == NULL) {
      fprintf(stderr, "DT is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dt = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("NSTP")) == NULL) {
      fprintf(stderr, "NSTP is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nstp = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("NPAS")) == NULL) {
      fprintf(stderr, "NPAS is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Npas = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("NRUN")) == NULL) {
      fprintf(stderr, "NRUN is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nrun = atol(cfg_tmp);

   if((cfg_tmp = cfg_read("GAMMA")) == NULL) {
      fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vgamma = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("NU")) == NULL) {
      fprintf(stderr, "NU is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vnu = atof(cfg_tmp);

   output = cfg_read("OUTPUT");
   rmsout = cfg_read("RMSOUT");
   initout = cfg_read("INITOUT");
   Nstpout = cfg_read("NSTPOUT");
   Npasout = cfg_read("NPASOUT");
   Nrunout = cfg_read("NRUNOUT");

   if((initout != NULL) || (Nstpout != NULL)  || (Npasout != NULL) || (Nrunout != NULL)) {
      if((cfg_tmp = cfg_read("OUTSTPX")) == NULL) {
         fprintf(stderr, "OUTSTPX is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpx = atol(cfg_tmp);

      if((cfg_tmp = cfg_read("OUTSTPY")) == NULL) {
         fprintf(stderr, "OUTSTPY is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpy = atol(cfg_tmp);
   }

   return;
}

/**
 *    Initialization of the space mesh, the potential, and the initial wave
 *    function.
 *    psi - array with the wave function values
 */
void init(double **psi) {
   long cnti, cntj;
   double gamma2, nu2;
   double tmp, cpsi1;

   if (opt == 1) par = 1.;
   else if (opt == 2) par = 2.;
   else{
      fprintf(stderr, "OPTION is not well defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }

   gamma2 = vgamma * vgamma;
   nu2 = vnu * vnu;

   Nx2 = Nx / 2; Ny2 = Ny / 2;
   dx2 = dx * dx; dy2 = dy * dy;

   cpsi1 = sqrt(pi * sqrt(1. / (vgamma * vnu)));

   for(cnti = 0; cnti < Nx; cnti ++) {
      x[cnti] = (cnti - Nx2) * dx;
      x2[cnti] = x[cnti] * x[cnti];
      for(cntj = 0; cntj < Ny; cntj ++) {
         y[cntj] = (cntj - Ny2) * dy;
         y2[cntj] = y[cntj] * y[cntj];

         pot[cnti][cntj] = (gamma2 * x2[cnti] + nu2 * y2[cntj]);
         tmp = exp(- 0.5 * (vgamma * x2[cnti] + vnu * y2[cntj]));
         psi[cnti][cntj] = tmp / cpsi1;
      }
   }

   return;
}

/**
 *    Crank-Nicolson scheme coefficients generation.
 */
void gencoef(void) {
   long cnti;

   Ax0 = 1. + dt / dx2;
   Ay0 = 1. + dt / dy2;

   Ax0r = 1. - dt / dx2;
   Ay0r = 1. - dt / dy2;

   Ax = - 0.5 * dt / dx2;
   Ay = - 0.5 * dt / dy2;

   calphax[Nx - 2] = 0.;
   cgammax[Nx - 2] = - 1. / Ax0;
   for (cnti = Nx - 2; cnti > 0; cnti --) {
      calphax[cnti - 1] = Ax * cgammax[cnti];
      cgammax[cnti - 1] = - 1. / (Ax0 + Ax * calphax[cnti - 1]);
   }

   calphay[Ny - 2] = 0.;
   cgammay[Ny - 2] = - 1. / Ay0;
   for (cnti = Ny - 2; cnti > 0; cnti --) {
      calphay[cnti - 1] = Ay * cgammay[cnti];
      cgammay[cnti - 1] = - 1. / (Ay0 + Ay * calphay[cnti - 1]);
   }

   return;
}

/**
 *    Calculation of the wave function norm and normalization.
 *    norm - wave function norm
 *    psi  - array with the wave function values
 *    tmpx - temporary array
 *    tmpy - temporary array
 */
void calcnorm(double *norm, double **psi, double **tmpx, double **tmpy) {
   int threadid;
   long cnti, cntj;

   #pragma omp parallel private(threadid, cnti, cntj)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            tmpy[threadid][cntj] = psi[cnti][cntj] * psi[cnti][cntj];
         }
         tmpx[0][cnti] = simpint(dy, tmpy[threadid], Ny);
      }
      #pragma omp barrier

      #pragma omp single
      *norm = sqrt(simpint(dx, tmpx[0], Nx));

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            psi[cnti][cntj] /= *norm;
         }
      }
   }

   return;
}


void calcmax(double *psi2m, double **psi) {
   double tmp;
   long cnti, cntj;

   *psi2m = 0.;

   #pragma omp for
   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         tmp = psi[cnti][cntj];
         tmp *= tmp;
         if(tmp > *psi2m) *psi2m = tmp;
      }
   }

   return;
}

/**
 *    Calculation of the chemical potential and energy.
 *    mu    - chemical potential
 *    en    - energy
 *    psi   - array with the wave function values
 *    dpsix - temporary array
 *    dpsiy - temporary array
 *    tmpxi - temporary array
 *    tmpyi - temporary array
 *    tmpxj - temporary array
 *    tmpyj - temporary array
 */
void calcmuen(double *mu, double *en, double **psi, double **dpsix, double **dpsiy, double **tmpxi, double **tmpyi, double **tmpxj, double **tmpyj) {
   int threadid;
   long cnti, cntj;
   double psi2, psi2lin, dpsi2;

   #pragma omp parallel private(threadid, cnti, cntj, psi2, psi2lin, dpsi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cntj = 0; cntj < Ny; cntj ++) {
         for(cnti = 0; cnti < Nx; cnti ++) {
            tmpxi[threadid][cnti] = psi[cnti][cntj];
         }
         diff(dx, tmpxi[threadid], tmpxj[threadid], Nx);
         for(cnti = 0; cnti < Nx; cnti ++) {
            dpsix[cnti][cntj] = tmpxj[threadid][cnti];
         }
      }

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            tmpyi[threadid][cntj] = psi[cnti][cntj];
         }
         diff(dy, tmpyi[threadid], tmpyj[threadid], Ny);
         for(cntj = 0; cntj < Ny; cntj ++) {
            dpsiy[cnti][cntj] = tmpyj[threadid][cntj];
         }
      }
      #pragma omp barrier

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            psi2 = psi[cnti][cntj] * psi[cnti][cntj];
            psi2lin = psi2 * G;
            dpsi2 = dpsix[cnti][cntj] * dpsix[cnti][cntj] +
                    dpsiy[cnti][cntj] * dpsiy[cnti][cntj];
            tmpyi[threadid][cntj] = (pot[cnti][cntj] + psi2lin) * psi2 + dpsi2;
            tmpyj[threadid][cntj] = (pot[cnti][cntj] + 0.5 * psi2lin) * psi2 + dpsi2;
         }
         tmpxi[0][cnti] = simpint(dy, tmpyi[threadid], Ny);
         tmpxj[0][cnti] = simpint(dy, tmpyj[threadid], Ny);
      }
   }

   *mu = simpint(dx, tmpxi[0], Nx);
   *en = simpint(dx, tmpxj[0], Nx);

   return;
}

/**
 *    Calculation of the root mean square radius.
 *    rms  - root mean square radius
 *    psi  - array with the wave function values
 *    tmpxi - temporary array
 *    tmpyi - temporary array
 *    tmpxj - temporary array
 *    tmpyj - temporary array
 */
void calcrms(double *rms, double **psi, double **tmpxi, double **tmpxj, double **tmpyi, double **tmpyj) {
   int threadid;
   long cnti, cntj;
   double tmp, psi2;

   #pragma omp parallel private(threadid, cnti, cntj, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cntj = 0; cntj < Ny; cntj ++) {
        for(cnti = 0; cnti < Nx; cnti ++) {
            psi2 = psi[cnti][cntj]*psi[cnti][cntj];
            tmpxi[threadid][cnti] = x[cnti] * psi2;
            tmpxj[threadid][cnti] = x2[cnti] * psi2;
         }
         tmpyi[0][cntj] = simpint(dx, tmpxi[threadid], Nx);
         tmpyj[0][cntj] = simpint(dx, tmpxj[threadid], Nx);
      }
      rms[1] = simpint(dy, tmpyi[0], Ny);
      tmp = simpint(dy, tmpyj[0], Ny);
      rms[2] = sqrt(tmp - rms[1] * rms[1]);
      #pragma omp barrier

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            psi2 = psi[cnti][cntj]*psi[cnti][cntj];
            tmpyi[threadid][cntj] = y[cntj] * psi2;
            tmpyj[threadid][cntj] = y2[cntj] * psi2;
         }
         tmpxi[0][cnti] = simpint(dy, tmpyi[threadid], Ny);
         tmpxj[0][cnti] = simpint(dy, tmpyj[threadid], Ny);
      }
      rms[3] = simpint(dx, tmpxi[0], Nx);
      tmp = simpint(dx, tmpxj[0], Nx);
      rms[4] = sqrt(tmp - rms[3] * rms[3]);
      #pragma omp barrier
   }
   rms[0] = sqrt(rms[2] * rms[2] + rms[4] * rms[4]);
   return;
}


/**
 *    Time propagation with respect to H1 (part of the Hamiltonian without spatial
 *    derivatives).
 *    psi - array with the wave function values
 */
void calcnu(double **psi) {
   long cnti, cntj;
   double psi2, psi2lin, tmp;

   #pragma omp parallel for private(cnti, cntj, psi2, psi2lin, tmp)
   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         psi2 = psi[cnti][cntj] * psi[cnti][cntj];
         psi2lin = psi2 * G;
         tmp = dt * (pot[cnti][cntj] + psi2lin);
         psi[cnti][cntj] *= exp(- tmp);
      }
   }

   return;
}

/**
 *    Time propagation with respect to H2 (x-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calclux(double **psi, double **cbeta) {
   int threadid;
   long cnti, cntj;
   double c;

   #pragma omp parallel private(threadid, cnti, cntj, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cntj = 0; cntj < Ny; cntj ++) {
         cbeta[threadid][Nx - 2] = psi[Nx - 1][cntj];
         for (cnti = Nx - 2; cnti > 0; cnti --) {
            c = - Ax * psi[cnti + 1][cntj] + Ax0r * psi[cnti][cntj] - Ax * psi[cnti - 1][cntj];
            cbeta[threadid][cnti - 1] =  cgammax[cnti] * (Ax * cbeta[threadid][cnti] - c);
         }
         psi[0][cntj] = 0.;
         for (cnti = 0; cnti < Nx - 2; cnti ++) {
            psi[cnti + 1][cntj] = calphax[cnti] * psi[cnti][cntj] + cbeta[threadid][cnti];
         }
         psi[Nx - 1][cntj] = 0.;
      }
   }

   return;
}

/**
 *    Time propagation with respect to H3 (y-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calcluy(double **psi, double **cbeta) {
   int threadid;
   long cnti, cntj;
   double c;

   #pragma omp parallel private(threadid, cnti, cntj, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         cbeta[threadid][Ny - 2] = psi[cnti][Ny - 1];
         for (cntj = Ny - 2; cntj > 0; cntj --) {
            c = - Ay * psi[cnti][cntj + 1] + Ay0r * psi[cnti][cntj] - Ay * psi[cnti][cntj - 1];
            cbeta[threadid][cntj - 1] =  cgammay[cntj] * (Ay * cbeta[threadid][cntj] - c);
         }
         psi[cnti][0] = 0.;
         for (cntj = 0; cntj < Ny - 2; cntj ++) {
            psi[cnti][cntj + 1] = calphay[cntj] * psi[cnti][cntj] + cbeta[threadid][cntj];
         }
         psi[cnti][Ny - 1] = 0.;
      }
   }

   return;
}

void outpsi2xy(double **psi, FILE *file) {
   long cnti, cntj;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntj = 0; cntj < Ny; cntj += outstpy) {
         fprintf(file, "%8le %8le %8le\n", x[cnti], y[cntj], psi[cnti][cntj] * psi[cnti][cntj]);
      }
      fprintf(file, "\n");
      fflush(file);
   }

   return;
}

void outdeny(double **psi, double *tmp, FILE *file) {
   long cnti, cntj;

    for(cntj = 0; cntj < Ny; cntj += outstpy) {
        for(cnti = 0; cnti < Nx; cnti ++) {
             tmp[cnti] = psi[cnti][cntj] * psi[cnti][cntj];
       }
       fprintf(file, "%8le %8le\n", y[cntj], simpint(dx, tmp, Nx));
       fflush(file);
    }
}

void outdenx(double **psi, double *tmp, FILE *file) {
   long cnti, cntj;

    for(cnti = 0; cnti < Nx; cnti += outstpx) {
       for(cntj = 0; cntj < Ny; cntj ++) {
             tmp[cntj] = psi[cnti][cntj] * psi[cnti][cntj];
       }
       fprintf(file, "%8le %8le\n", x[cnti], simpint(dy, tmp, Ny));
       fflush(file);
    }
}
