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
 *    This program solves the time-dependent Gross–Pitaevskii nonlinear
 *    partial differential equation in two space dimensions in anisotropic
 *    trap using the real-time propagation. The Gross–Pitaevskii equation
 *    describes the properties of a dilute trapped Bose–Einstein condensate.
 *    The equation is solved using the split-step Crank–Nicolson method by
 *    discretizing space and time. The discretized equation is then propagated
 *    in real time over small time steps.
 *
 *    Description of variables used in the code:
 *
 *    opt     - decides which rescaling of GP equation will be used
 *    par     - parameter for rescaling of GP equation
 *    psi     - array with the wave function values
 *    pot     - array with the values of the potential
 *    g_2d    - final nonlinearity
 *    Gpar    - coefficient that multiplies nonlinear term in non-stationary
 *              problem during final Nrun iterations
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
 *    Nstp    - number of subsequent iterations with fixed nonlinearity g_2d
 *    Npas    - number of subsequent iterations with the fixed nonlinearity g_2d
 *    Nrun    - number of final iterations with the fixed nonlinearity g_2d
 *    output  - output file with the summary of values of all physical quantities
 *    rmsout  - output file with the summary of values of RMS quantities
 *    initout - output file with the initial wave function
 *    Nstpout - output file with the wave function obtained after the first
 *              Nstp iterations
 *    Npasout - output file with the wave function obtained after the
 *              subsequent Npas iterations, with the fixed nonlinearity g_2d
 *    Nrunout - output file with the final wave function obtained after the
 *              final Nrun iterations
 *    dyn  - output file with the time dependence of RMS during the final
 *              Nrun iterations
 *    outstpx - discretization step in the x-direction used to save wave
 *              functions
 *    outstpy - discretization step in the y-direction used to save wave
 *              functions
 *    outstpt - time discretization step used to save RMS of the wave function
 */

#include "real2d.h"

int main(int argc, char **argv) {
   FILE *out;
   FILE *file;
   FILE *filerms;
   FILE *dyna;
   char filename[MAX_FILENAME_SIZE];
   int nthreads;
   long cnti, cntj, cntk;
   double norm, ek, ev, ei;
   double tt, vgammat, vnut, vgamma2t, vnu2t;
   double *rms;
   double complex **psi;
   double **dpsix, **dpsiy;
   double **tmpxi, **tmpyi, **tmpxj, **tmpyj, **tmpxk, **tmpyk, **tmpxl, **tmpyl;
   double *tmpx, *tmpy;
   double psi2;
   double **abc;

   time_t clock_beg, clock_end;
   clock_beg = time(NULL);

   if((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) {
      fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
      exit(EXIT_FAILURE);
   }

   if(! cfg_init(argv[2])) {
      fprintf(stderr, "Wrong input parameter file.\n");
      exit(EXIT_FAILURE);
   }

   pi = 4. * atan(1.);

   readpar();

   #pragma omp parallel
      #pragma omp master
         nthreads = omp_get_num_threads();

   rms = alloc_double_vector(RMS_ARRAY_SIZE);
   x = alloc_double_vector(Nx);
   y = alloc_double_vector(Ny);
   kx = alloc_double_vector(Nx);
   ky = alloc_double_vector(Ny);

   x2 = alloc_double_vector(Nx);
   y2 = alloc_double_vector(Ny);
   kx2 = alloc_double_vector(Nx);
   ky2 = alloc_double_vector(Ny);

   pot = alloc_double_matrix(Nx, Ny);
   psi = alloc_complex_matrix(Nx, Ny);
   abc = alloc_double_matrix(Nx, Ny);
   um1 = alloc_complex_matrix(Nx, Ny);

   dpsix = alloc_double_matrix(Nx, Ny);
   dpsiy = alloc_double_matrix(Nx, Ny);

   tmpxi = alloc_double_matrix(nthreads, Nx);
   tmpyi = alloc_double_matrix(nthreads, Ny);
   tmpxj = alloc_double_matrix(nthreads, Nx);
   tmpyj = alloc_double_matrix(nthreads, Ny);
   tmpxk = alloc_double_matrix(nthreads, Nx);
   tmpyk = alloc_double_matrix(nthreads, Ny);
   tmpxl = alloc_double_matrix(nthreads, Nx);
   tmpyl = alloc_double_matrix(nthreads, Ny);

   tmpx = alloc_double_vector(Nx);
   tmpy = alloc_double_vector(Ny);

   /* allocate space and initialize real array to transform */
   fftin = (fftw_complex *)fftw_malloc( sizeof(fftw_complex)*Nx*Ny);
   fftout = (fftw_complex *)fftw_malloc( sizeof(fftw_complex)*Nx*Ny);

   fftw_init_threads();
   fftw_plan_with_nthreads(nthreads);
   pf = fftw_plan_dft_2d(Nx, Ny, fftin, fftout, FFTW_FORWARD, FFT_FLAG );
   pb = fftw_plan_dft_2d(Nx, Ny, fftin, fftout, FFTW_BACKWARD, FFT_FLAG );

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

   fprintf(out, " Real time propagation 2d,   OPTION = %d\n\n", opt);
   fprintf(out, "  Unit of length AHO = %10.8le\n", aho);
   fprintf(out, "  Condensate number N0 = %li\n", N0);
   fprintf(out, "  Scattering length a0 = %.4f*aB\n", as);
   fprintf(out, "  Nonlinearity G0_3D = %.7f, G0_2D = %.7f\n", g0_3d, g0_2d);
   fprintf(out, "  Parameters of trap: GAMMA = %.4f, NU = %.4f\n", vgamma, vnu);
   fprintf(out, "  Parameters of oscillation: AMP = %.4f, OMEGA = %.4f\n", amp, freq);
   fprintf(out, " # Space Stp: NX = %li, NY = %li\n", Nx,Ny);
   fprintf(out, "  Space Step: DX = %.4f, DY = %.4f\n", dx,dy);
   fprintf(out, " # Time Stp : NPAS = %li, NRUN = %li, NSTP = %li\n", Npas, Nrun, Nstp);
   fprintf(out, "   Time Step: DT = %.6f\n\n",  dt);
   fprintf(out, " * Change for dynamics: GPAR = %.3f *\n\n", Gpar);
   fprintf(out, "                  -------------------------------------------------------------------------\n");
   fprintf(out, "                    Norm      Kin/N      Pot/N      Int/N      <rho>      |Psi(0,0)|^2\n");
   fprintf(out, "                  -------------------------------------------------------------------------\n");
   fflush(out);
   if(rmsout != NULL) {
     fprintf(filerms, " Real time propagation 2d,   OPTION = %d\n\n", opt);
     fprintf(filerms, "                  ------------------------------------------------------------\n");
      fprintf(filerms, "Values of rms size:    <x>         <x2>        <y>         <y2>        <rho>\n");
     fprintf(filerms, "                  ------------------------------------------------------------\n");
     fflush(filerms);
   }

   printf("initialization!\n");

   initpsi(psi, abc);
   G = par * g0_2d;

   grad2(um1);

   calcnorm(&norm, psi, tmpxi, tmpyi);
   calcmuen(&ek, &ev, &ei, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj, tmpxk, tmpyk, tmpxl, tmpyl);
   calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
   calcmax(&psi2, psi);

   fprintf(out, "Initial : %15.4f %11.3f %11.3f %11.4f %10.3f %10.6f\n", norm, ek / par, ev / par, ei / par, *rms, psi2);
   fflush(out);
   if(rmsout != NULL) {
      fprintf(filerms, "           Initial:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
      fflush(filerms);
   }
   if(initout != NULL) {
      sprintf(filename, "%s.txt", initout);
      file = fopen(filename, "w");
      outpsixy(psi, file);
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

   if(dynaout != NULL) {
      sprintf(filename, "%s.txt", dynaout);
      dyna = fopen(filename, "w");
   }
   else dyna = NULL;

   if(Npas != 0){
      printf("%ld step single component evolution\n", Npas);
      for(cntk = 1; cntk <= Npas; cntk ++){

         tt = cntk * dt * par;
         vgammat = vgamma * (1. + amp * sin(freq * tt));
         vnut = vnu * (1. + amp * sin(freq * tt));
         vgamma2t = vgammat * vgammat;
         vnu2t = vnut * vnut;

         for(cnti = 0; cnti < Nx; cnti ++) {
            for(cntj = 0; cntj < Ny; cntj ++) {
               pot[cnti][cntj] = (vgamma2t * x2[cnti] + vnu2t * y2[cntj]) ;
            }
         }

         calcnu(psi);

         // Fourier transform
         calcfft(psi, fftin, fftout, pf);
         // Evolution in momentum space
         #pragma omp parallel for private(cnti, cntj)
         for(cnti = 0; cnti < Nx; cnti++){
            for(cntj = 0; cntj < Ny; cntj++){
               psi[cnti][cntj] *= um1[cnti][cntj];
            }
         }
         // Inverse Fourier transform
         calcfft(psi, fftin, fftout, pb);

         calcnu(psi);
         // pdcon(psi);

         printf("%ld\n", cntk);
         if((dynaout != NULL) && (cntk % outstpt == 0)) {
            calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
            calcmax(&psi2, psi);
            // Modified Sep. 5
            // add monitor of the energies
            calcmuen(&ek, &ev, &ei, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj, tmpxk, tmpyk, tmpxl, tmpyl);
            fprintf(dyna, "%8le   %8le   %8le   %8le   %8le   %8le   %8le   %8le   %8le\n", cntk * dt * par, psi2, rms[1], rms[2], rms[3], rms[4], ek / par, ev / par, ei / par);
            fflush(dyna);
         }
         if((tempout != NULL) && (cntk % outdenstp == 0)){
            sprintf(filename, "%s_%li.txt", tempout, (long) (cntk * dt * par));
            file = fopen(filename, "w");
            outpsixy(psi, file);
            fclose(file);
         }
      }
      calcnorm(&norm, psi, tmpxi, tmpyi);
      calcmuen(&ek, &ev, &ei, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj, tmpxk, tmpyk, tmpxl, tmpyl);
      calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
      calcmax(&psi2, psi);
      fprintf(out, "After NPAS iter.:%8.4f %11.3f %11.3f %11.4f %10.3f %10.6f\n", norm, ek / par, ev / par, ei / par, *rms, psi2);
      fflush(out);
      if(rmsout != NULL) {
         fprintf(filerms, "  After NPAS iter.:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
         fflush(filerms);
      }
      if(Npasout != NULL) {
         sprintf(filename, "%s.txt", Npasout);
         file = fopen(filename, "w");
         outpsixy(psi, file);
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

   if(Nrun != 0){
      printf("%ld step evolution of varied interaction\n", Nrun);
      for(cntk = 1; cntk <= Nrun; cntk ++) {

         calcnu(psi);

         calcfft(psi, fftin, fftout, pf);   // Fourier transform
         // Output the momentum distribution
         if((tempmom != NULL) && (cntk % outdenstp == 0)){
            sprintf(filename, "%s_%li.txt", tempmom, (long)(cntk * dt * par));
            file = fopen(filename, "w");
            outpsixy(psi, file);
            fclose(file);
         }
         // Evolution in momentum space
         #pragma omp parallel for private(cnti, cntj)
         for(cnti = 0; cnti < Nx; cnti++){
            for(cntj = 0; cntj < Ny; cntj++){
               psi[cnti][cntj] *= um1[cnti][cntj];
            }
         }
         // Inverse Fourier transform
         calcfft(psi, fftin, fftout, pb);

         calcnu(psi);
         // pdcon(psi);
         printf("%ld\n", cntk);
         if((dynaout != NULL) && ((cntk + Npas) % outstpt == 0)) {
            calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
            calcmax(&psi2, psi);
            calcmuen(&ek, &ev, &ei, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj, tmpxk, tmpyk, tmpxl, tmpyl);
            fprintf(dyna, "%8le   %8le   %8le   %8le   %8le   %8le   %8le   %8le   %8le\n", (cntk + Npas) * dt * par, psi2, rms[1], rms[2], rms[3], rms[4], ek / par, ev / par, ei / par);
            fflush(dyna);
         }
         if((tempout != NULL) && ((cntk + Npas)  % outdenstp == 0)){
            sprintf(filename, "%s_%li.txt", tempout, (long)((cntk + Npas) * dt * par));
            file = fopen(filename, "w");
            outpsixy(psi, file);
            fclose(file);
         }
      }

      calcnorm(&norm, psi, tmpxi, tmpyi);
      calcmuen(&ek, &ev, &ei, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj, tmpxk, tmpyk, tmpxl, tmpyl);
      calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
      calcmax(&psi2, psi);
      fprintf(out, "After NRUN iter.:%8.4f %11.3f %11.3f %11.4f %10.3f %10.6f\n", norm, ek / par, ev / par, ei / par, *rms, psi2);
      fflush(out);
      if(rmsout != NULL) {
         fprintf(filerms, "  After NRUN iter.:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
         fflush(filerms);
      }
      if(Nrunout != NULL) {
         sprintf(filename, "%s.txt", Nrunout);
         file = fopen(filename, "w");
         outpsixy(psi, file);
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

   G = g0_2d * par;

   if(Nstp != 0){
      printf("%ld step single component evolution\n", Nstp);
      for(cntk = 1; cntk <= Nstp; cntk ++){
         calcnu(psi);

         calcfft(psi, fftin, fftout, pf);   // Fourier transform
         // Output the momentum distribution
         if((tempmom != NULL) && ((cntk + Nrun) % outdenstp == 0)){
            sprintf(filename, "%s_%li.txt", tempmom, (long)((cntk + Nrun)* dt * par));
            file = fopen(filename, "w");
            outpsixy(psi, file);
            fclose(file);
         }
         // Evolution in momentum space
         #pragma omp parallel for private(cnti, cntj)
         for(cnti = 0; cnti < Nx; cnti++){
            for(cntj = 0; cntj < Ny; cntj++){
               psi[cnti][cntj] *= um1[cnti][cntj];
            }
         }
         // Inverse Fourier transform
         calcfft(psi, fftin, fftout, pb);

         calcnu(psi);
         // pdcon(psi);

         printf("%ld\n", cntk + Nrun);
         if((dynaout != NULL) && ((cntk + Nrun) % outstpt == 0)) {
            calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
            calcmax(&psi2, psi);
            // Modified Sep. 5
            // add monitor of the energies
            calcmuen(&ek, &ev, &ei, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj, tmpxk, tmpyk, tmpxl, tmpyl);
            fprintf(dyna, "%8le   %8le   %8le   %8le   %8le   %8le   %8le   %8le   %8le\n", (cntk + Nrun + Npas) * dt * par, psi2, rms[1], rms[2], rms[3], rms[4], ek / par, ev / par, ei / par);
            fflush(dyna);
         }
         if((tempout != NULL) && ((cntk + Nrun) % outdenstp == 0)){
            sprintf(filename, "%s_%li.txt", tempout, (long) ((cntk + Nrun) * dt * par));
            file = fopen(filename, "w");
            outpsixy(psi, file);
            fclose(file);
         }
      }
      calcnorm(&norm, psi, tmpxi, tmpyi);
      calcmuen(&ek, &ev, &ei, psi, dpsix, dpsiy, tmpxi, tmpyi, tmpxj, tmpyj, tmpxk, tmpyk, tmpxl, tmpyl);
      calcrms(rms, psi, tmpxi, tmpxj, tmpyi, tmpyj);
      calcmax(&psi2, psi);
      fprintf(out, "After NSTP iter.:%8.4f %11.3f %11.3f %11.4f %10.3f %10.6f\n", norm, ek / par, ev / par, ei / par, *rms, psi2);
      fflush(out);
      if(rmsout != NULL) {
         fprintf(filerms, "  After NSTP iter.:%10.5f   %9.5f   %9.5f   %9.5f   %9.5f\n", rms[1], rms[2], rms[3], rms[4], rms[0]);
         fflush(filerms);
      }
      if(Nstpout != NULL) {
         sprintf(filename, "%s.txt", Nstpout);
         file = fopen(filename, "w");
         outpsixy(psi, file);
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
   }
   if(dynaout != NULL) fclose(dyna);
   if(rmsout != NULL) {
      fprintf(filerms, "                  ------------------------------------------------------------\n");
      fflush(filerms);
      fclose(filerms);
   }

   fprintf(out, "                  ---------------------------------------------------------------------------\n\n");

   free_double_vector(rms);

   free_double_vector(x);
   free_double_vector(y);

   free_double_vector(x2);
   free_double_vector(y2);

   free_double_vector(kx);
   free_double_vector(ky);

   free_double_vector(kx2);
   free_double_vector(ky2);

   free_double_matrix(pot);
   free_complex_matrix(psi);
   free_complex_matrix(um1);
   free_double_matrix(abc);

   free_double_matrix(dpsix);
   free_double_matrix(dpsiy);

   free_double_matrix(tmpxi);
   free_double_matrix(tmpyi);
   free_double_matrix(tmpxj);
   free_double_matrix(tmpyj);

   free_double_vector(tmpx);
   free_double_vector(tmpy);

   fftw_destroy_plan(pf);
   fftw_destroy_plan(pb);

   fftw_free(fftin);
   fftw_free(fftout);

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
      N0 = atol(cfg_tmp);


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

      g0_3d = 4. * (4. * atan(1.)) * as * N0 * BOHR_RADIUS / aho;
      g0_2d = g0_3d / (sqrt(2. * (4. * atan(1.))) * d_z);

   } else {
      g0_2d = atof(cfg_tmp);
      g0_3d = g0_2d * (sqrt(2. * (4. * atan(1.))) * d_z);
   }

   if((cfg_tmp = cfg_read("GPAR")) == NULL) {
      fprintf(stderr, "GPAR is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Gpar = atof(cfg_tmp);

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

   if((cfg_tmp = cfg_read("AMP")) == NULL) {
      fprintf(stderr, "AMP is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   amp = atof(cfg_tmp);

   if((cfg_tmp = cfg_read("FREQ")) == NULL) {
      fprintf(stderr, "FREQ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   freq = atof(cfg_tmp);

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

   output = cfg_read("OUTPUT");
   initout = cfg_read("INITOUT");
   rmsout = cfg_read("RMSOUT");
   dynaout = cfg_read("DYNAOUT");
   Nstpout = cfg_read("NSTPOUT");
   Npasout = cfg_read("NPASOUT");
   Nrunout = cfg_read("NRUNOUT");
   tempout = cfg_read("TEMPOUT");
   tempmom = cfg_read("TEMPMOM");

   if((initout != NULL) || (Nstpout != NULL) || (Npasout != NULL) || (Nrunout != NULL) || (tempout != NULL) || (tempmom != NULL)) {
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

   if((tempout != NULL) || (tempmom != NULL)) {
      if((cfg_tmp = cfg_read("OUTDENSTP")) == NULL) {
         fprintf(stderr, "OUTDENSTP is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outdenstp = atol(cfg_tmp);
   }

   if(dynaout != NULL) {
      if((cfg_tmp = cfg_read("OUTSTPT")) == NULL) {
         fprintf(stderr, "OUTSTPT is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpt = atol(cfg_tmp);
   }

   return;
}

/**
 *    Initialization of the space mesh, the potential, and the initial wave
 *    function.
 *    psi - array with the wave function values
 */
void initpsi(double complex **psi, double **abc) {
   long cnti, cntj;
   double vgamma2, vnu2;
   double cpsi, tmp;
   FILE *file;

   if (opt == 1) par = 1.;
   else if (opt == 2) par = 2.;
   else{
      fprintf(stderr, "OPTION is not well defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }

   if (opt == 1) par = 1.;
   else if (opt == 2) par = 2.;
   else{
      fprintf(stderr, "OPTION is not well defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }

   vgamma2 = vgamma * vgamma;
   vnu2 = vnu * vnu;

   Nx2 = Nx / 2; Ny2 = Ny / 2;
   dx2 = dx * dx; dy2 = dy * dy;

   for(cnti = 0; cnti < Nx; cnti ++) {
      x[cnti] = (cnti - Nx2) * dx;
      x2[cnti] = x[cnti] * x[cnti];
   }

   for(cntj = 0; cntj < Ny; cntj ++) {
      y[cntj] = (cntj - Ny2) * dy;
      y2[cntj] = y[cntj] * y[cntj];
   }

   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         pot[cnti][cntj] = (vgamma2 * x2[cnti] + vnu2 * y2[cntj]);
      }
   }

   if(Nstp!=0) {
      cpsi = sqrt(pi * sqrt(1. / (vgamma * vnu)));
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            tmp = exp(- 0.5 * (vgamma * x2[cnti] + vnu * y2[cntj]));
            psi[cnti][cntj] = tmp / cpsi;
         }
      }
   }

   if((file = fopen("imag2d-den.txt", "r"))==NULL) {       /* open a text file for reading */
      printf("Run the program using the input file to read. i.g.: imag2d-den.txt\n");
      exit(1);               /*couldn't open the requested file!*/
   }
   file = fopen("imag2d-den.txt", "r");
   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         if(fscanf(file,"%lf %lf %lf\n", &x[cnti], &y[cntj], &abc[cnti][cntj])) {
//             printf("%8le %8le %8le\n", x[cnti], y[cntj], abc[cnti][cntj]);
         }
      }
   }
   fclose(file);

   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         psi[cnti][cntj] = sqrt(abc[cnti][cntj]);
      }
   }
   return;
}

double randn(double mu, double sigma){
   double U1, U2, W, mult;
   static double X1, X2;
   static int call = 0;

   if (call == 1){
      call = !call;
      return (mu + sigma * (double) X2);
   }

   do{
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
   }
   while (W >= 1 || W == 0);

   mult = sqrt ((-2 * log (W)) / W);
   X1 = U1 * mult;
   X2 = U2 * mult;

   call = !call;

   return (mu + sigma * (double) X1);
}

/**
 *    Calculation of the wave function norm and normalization.
 *    norm - wave function norm
 *    psi  - array with the wave function values
 *    tmpx - temporary array
 *    tmpy - temporary array
 */
void calcnorm(double *norm, double complex **psi, double **tmpx, double **tmpy) {
   int threadid;
   long cnti, cntj;

   #pragma omp parallel private(threadid, cnti, cntj)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            tmpy[threadid][cntj] = cabs(psi[cnti][cntj]);
            tmpy[threadid][cntj] *= tmpy[threadid][cntj];
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


void calcmax(double *psi2m, double complex **psi) {
   double tmp;
   long cnti, cntj;

   *psi2m = 0.;

   #pragma omp for
   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         tmp = cabs(psi[cnti][cntj]);
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
void calcmuen(double *ek, double *ev, double *ei, double complex **psi, double **dpsix, double **dpsiy, double **tmpxi, double **tmpyi, double **tmpxj, double **tmpyj, double **tmpxk, double **tmpyk, double **tmpxl, double **tmpyl) {
   int threadid;
   long cnti, cntj;
   double psi2, psi2lin, dpsi2;

 // Modified on Sep 5.
 // Correct the evaluation of the kinetic energy.

   #pragma omp parallel private(threadid, cnti, cntj, psi2, psi2lin, dpsi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cntj = 0; cntj < Ny; cntj ++) {
         for(cnti = 0; cnti < Nx; cnti ++) {
            tmpxi[threadid][cnti] = creal(psi[cnti][cntj]);
            tmpxk[threadid][cnti] = cimag(psi[cnti][cntj]);
         }
         diff(dx, tmpxi[threadid], tmpxj[threadid], Nx);
         diff(dx, tmpxk[threadid], tmpxl[threadid], Nx);
         for(cnti = 0; cnti < Nx; cnti ++) {
            dpsix[cnti][cntj] = tmpxj[threadid][cnti] * tmpxj[threadid][cnti]
            + tmpxl[threadid][cnti] * tmpxl[threadid][cnti];
         }
      }
      #pragma omp barrier

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            tmpyi[threadid][cntj] = creal(psi[cnti][cntj]);
            tmpyk[threadid][cntj] = cimag(psi[cnti][cntj]);
         }
         diff(dy, tmpyi[threadid], tmpyj[threadid], Ny);
         diff(dy, tmpyk[threadid], tmpyl[threadid], Ny);
         for(cntj = 0; cntj < Ny; cntj ++) {
            dpsiy[cnti][cntj] = tmpyj[threadid][cntj] * tmpyj[threadid][cntj]
            + tmpyl[threadid][cntj] * tmpyl[threadid][cntj];
         }
      }
      #pragma omp barrier

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            psi2 = cabs(psi[cnti][cntj]);
            psi2 *= psi2;
            psi2lin = psi2 * G;
            // change the evaluation of chemical potential to kinetic energy
            dpsi2 = dpsix[cnti][cntj] + dpsiy[cnti][cntj];
            tmpyi[threadid][cntj] = dpsi2;
            tmpyj[threadid][cntj] = pot[cnti][cntj] * psi2;
            tmpyk[threadid][cntj] = psi2lin * psi2;
         }
         tmpxi[0][cnti] = simpint(dy, tmpyi[threadid], Ny);
         tmpxj[0][cnti] = simpint(dy, tmpyj[threadid], Ny);
         tmpxk[0][cnti] = simpint(dy, tmpyk[threadid], Ny);
      }
   }

   *ek = simpint(dx, tmpxi[0], Nx);
   *ev = simpint(dx, tmpxj[0], Nx);
   *ei = simpint(dx, tmpxk[0], Nx);

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
void calcrms(double *rms, double complex **psi, double **tmpxi, double **tmpxj, double **tmpyi, double **tmpyj) {
   int threadid;
   long cnti, cntj;
   double tmp, psi2;

   #pragma omp parallel private(threadid, cnti, cntj, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for(cntj = 0; cntj < Ny; cntj ++) {
        for(cnti = 0; cnti < Nx; cnti ++) {
            psi2 = cabs(psi[cnti][cntj]);
            psi2 *= psi2;
            tmpxi[threadid][cnti] = x[cnti] * psi2;
            tmpxj[threadid][cnti] = x2[cnti] * psi2;
         }
         tmpyi[0][cntj] = simpint(dx, tmpxi[threadid], Nx);
         tmpyj[0][cntj] = simpint(dx, tmpxj[threadid], Nx);
      }
      #pragma omp barrier

      #pragma omp single
      rms[1] = simpint(dy, tmpyi[0], Ny);
      #pragma omp single
      tmp = simpint(dy, tmpyj[0], Ny);
      #pragma omp single
      rms[2] = sqrt(tmp - rms[1] * rms[1]);

      #pragma omp for
      for(cnti = 0; cnti < Nx; cnti ++) {
         for(cntj = 0; cntj < Ny; cntj ++) {
            psi2 = cabs(psi[cnti][cntj]);
            psi2 *= psi2;
            tmpyi[threadid][cntj] = y[cntj] * psi2;
            tmpyj[threadid][cntj] = y2[cntj] * psi2;
         }
         tmpxi[0][cnti] = simpint(dy, tmpyi[threadid], Ny);
         tmpxj[0][cnti] = simpint(dy, tmpyj[threadid], Ny);
      }
      #pragma omp barrier

      #pragma omp single
      rms[3] = simpint(dx, tmpxi[0], Nx);
      #pragma omp single
      tmp = simpint(dx, tmpxj[0], Nx);
      #pragma omp single
      rms[4] = sqrt(tmp - rms[3] * rms[3]);
   }

   rms[0] = sqrt(rms[2] * rms[2] + rms[4] * rms[4]);
   return;
}

/**
 *    Time propagation with respect to H1 (part of the Hamiltonian without spatial
 *    derivatives).
 *    psi - array with the wave function values
 */
void calcnu(double complex **psi) {
   long cnti, cntj;
   double psi2, psi2lin, tmp;

   #pragma omp parallel for private(cnti, cntj, psi2, psi2lin, tmp)
   for(cnti = 0; cnti < Nx; cnti ++) {
      for(cntj = 0; cntj < Ny; cntj ++) {
         psi2 = cabs(psi[cnti][cntj]);
         psi2 *= psi2;
         psi2lin = psi2 * G;
         tmp = 0.5 * dt * (pot[cnti][cntj] + psi2lin);
         psi[cnti][cntj] *= cexp(- I * tmp);
      }
   }

   return;
}

void grad2(double complex **um1){
   double pi1 = 2 * pi / (double) Nx / dx;
   double pj1 = 2 * pi / (double) Ny / dy;
   double pi2 = -dt * pi1 * pi1;
   double pj2 = -dt * pj1 * pj1;
   double xyn = 1.0 / ( (double) Nx * (double) Ny );
   long i, j;
   long i1, j1;
   double pi3, pij3;

   #pragma omp for
   for(i = 0; i < Nx; i++){
      if(i < Nx/2){
         i1 = i;
      }
      else{
         i1 = i - Nx;
      }
      // i1 = i - Nx/2;
      pi3 = (double) i1 * (double) i1 * pi2;
      for(j = 0; j < Ny; j++){
         if(j < Ny/2){
            j1 = j;
         }
         else{
            j1 = j - Ny;
         }
         // j1 = j - Ny/2;
         pij3 = pi3 + (double) j1 * (double) j1 * pj2;
         um1[i][j] = xyn * cexp( I * pij3);
      }
   }
   return;
}

void calcfft(double complex **psi, fftw_complex *fftin, fftw_complex *fftout, fftw_plan p){
   long cnti, cntj;

   #pragma omp parallel for private(cnti, cntj)
   for(cnti = 0; cnti < Nx; cnti++){
      for(cntj = 0; cntj < Ny; cntj++){
         fftin[cnti * Ny + cntj][0] = creal(psi[cnti][cntj]);
         fftin[cnti * Ny + cntj][1] = cimag(psi[cnti][cntj]);
      }
   }

   fftw_execute(p);

   #pragma omp parallel for private(cnti, cntj)
   for(cnti = 0; cnti < Nx; cnti++){
      for(cntj = 0; cntj < Ny; cntj++){
         psi[cnti][cntj] = fftout[cnti * Ny + cntj][0] + I * fftout[cnti * Ny + cntj][1];
      }
   }

   return;
}

// void pdcon(double complex **psi){
//    long cnti, cntj;
//    #pragma omp parallel for private(cntj)
//    for (cntj = 0; cntj < Ny; cntj ++) {
//       psi[Nx-1][cntj] = psi[0][cntj];
//    }

//    #pragma omp parallel for private(cnti)
//    for (cnti = 0; cnti < Nx; cnti ++) {
//       psi[cnti-1][Ny] = psi[cnti][0];
//    }
// }

void outpsixy(double complex **psi, FILE *file) {
   long cnti, cntj;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntj = 0; cntj < Ny; cntj += outstpy) {
         fprintf(file, "%8le  %8le  %8le  %8le\n", x[cnti], y[cntj], creal(psi[cnti][cntj]), cimag(psi[cnti][cntj]));
      }
      fprintf(file, "\n");
      fflush(file);
   }

   return;
}

// void outpsip(double complex **psi, FILE *file) {
//    long cnti, cntj;

//    for(cnti = 0; cnti < Nx; cnti += outstpx) {
//       for(cntj = 0; cntj < Ny; cntj += outstpy) {
//          fprintf(file, "%8le  %8le  %8le   %8le\n", x[cnti], y[cntj], creal(psi[cnti][cntj]), cimag(psi[cnti][cntj]));
//       }
//       fprintf(file, "\n");
//       fflush(file);
//    }

//    return;
// }

void outdenx(double complex **psi, double *tmp, FILE *file) {
   long cnti, cntj;

   for(cnti = 0; cnti < Nx; cnti += outstpx) {
      for(cntj = 0; cntj < Ny; cntj ++) {
            tmp[cntj] = cabs(psi[cnti][cntj]) * cabs(psi[cnti][cntj]);
      }
      fprintf(file, "%8le %8le\n", x[cnti], simpint(dy, tmp, Ny));
      fflush(file);
   }
}

void outdeny(double complex **psi, double *tmp, FILE *file) {
   long cnti, cntj;

   for(cntj = 0; cntj < Ny; cntj += outstpy) {
       for(cnti = 0; cnti < Nx; cnti ++) {
            tmp[cnti] = cabs(psi[cnti][cntj]) * cabs(psi[cnti][cntj]);
      }
      fprintf(file, "%8le %8le\n", y[cntj], simpint(dx, tmp, Nx));
      fflush(file);
   }
}
