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
 *    Utility functions for integration (Simson's rule) and differentiation.
 */

#include "diffint.h"

/**
 *    Spatial 1D integration with Simpson's rule.
 *    h - space step
 *    f - array with the function values
 *    N - number of integration points
 */
double simpint(double h, double *f, long N) {
   long cnti;
   double sum, sumi, sumj, sumk;
   
   sumi = 0.; sumj = 0.; sumk = 0.;

   for(cnti = 1; cnti < N - 1; cnti += 2) {
      sumi += f[cnti];
      sumj += f[cnti - 1];
      sumk += f[cnti + 1];
   }
   
   sum = sumj + 4. * sumi + sumk;
   if(N % 2 == 0) sum += (5. * f[N - 1] + 8. * f[N - 2] - f[N - 3]) / 4.;

   return sum * h / 3.;
}

/**
 *    Richardson extrapolation formula for calculation of space
 *    derivatives.
 *    h  - space step
 *    f  - array with the function values
 *    df - array with the first derivatives of the function
 *    N  - number of space mesh points
 */
void diff(double h, double *f, double *df, long N) {
   long cnti;
   
   df[0] = 0.;
   df[1] = (f[2] - f[0]) / (2. * h);
   
   for(cnti = 2; cnti < N - 2; cnti ++) {
      df[cnti] = (f[cnti - 2] - 8. * f[cnti - 1] + 8. * f[cnti + 1] - f[cnti + 2]) / (12. * h); 
   }
   
   df[N - 2] = (f[N - 1] - f[N - 3]) / (2. * h);
   df[N - 1] = 0.;
   
   return;
}
