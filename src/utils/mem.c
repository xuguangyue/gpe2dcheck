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
 *    Utility functions for allocation/deallocation of memory.
 */

#include "mem.h"

/**
 *    Double vector allocation
 */
double *alloc_double_vector(long Nx) {
   double *vector;
   
   if((vector = (double *) malloc((size_t) (Nx * sizeof(double)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the vector.\n");
      exit(EXIT_FAILURE);  
   }
   
   return vector;
}

/**
 *    Complex vector allocation
 */
double complex *alloc_complex_vector(long Nx) {
   double complex *vector;
   
   if((vector = (double complex *) malloc((size_t) (Nx * sizeof(double complex)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the vector.\n");
      exit(EXIT_FAILURE);  
   }
   
   return vector;
}

/**
 *    Double matrix allocation
 */
double **alloc_double_matrix(long Nx, long Ny) {
   long cnti;
   double **matrix;
   
   if((matrix = (double **) malloc((size_t) (Nx * sizeof(double *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);  
   }
   if((matrix[0] = (double *) malloc((size_t) (Nx * Ny * sizeof(double)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);  
   }
   for(cnti = 1; cnti < Nx; cnti ++)
      matrix[cnti] = matrix[cnti - 1] + Ny;
   
   return matrix;
}

/**
 *    Complex matrix allocation
 */
double complex **alloc_complex_matrix(long Nx, long Ny) {
   long cnti;
   double complex **matrix;
   
   if((matrix = (double complex **) malloc((size_t) (Nx * sizeof(double complex *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);  
   }
   if((matrix[0] = (double complex *) malloc((size_t) (Nx * Ny * sizeof(double complex)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);  
   }
   for(cnti = 1; cnti < Nx; cnti ++)
      matrix[cnti] = matrix[cnti - 1] + Ny;
   
   return matrix;
}

/**
 *    Double tensor allocation
 */
double ***alloc_double_tensor(long Nx, long Ny, long Nz) {
   long cnti, cntj;
   double ***tensor;
   
   if((tensor = (double ***) malloc((size_t) (Nx * sizeof(double **)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0] = (double **) malloc((size_t) (Nx * Ny * sizeof(double *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0][0] = (double *) malloc((size_t) (Nx * Ny * Nz * sizeof(double)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   for(cntj = 1; cntj < Ny; cntj ++)
      tensor[0][cntj] = tensor[0][cntj-1] + Nz;
   for(cnti = 1; cnti < Nx; cnti ++) {
      tensor[cnti] = tensor[cnti - 1] + Ny;
      tensor[cnti][0] = tensor[cnti - 1][0] + Ny * Nz;
      for(cntj = 1; cntj < Ny; cntj ++)
         tensor[cnti][cntj] = tensor[cnti][cntj - 1] + Nz;
   }
   
   return tensor;
}

/**
 *    Complex tensor allocation
 */
double complex ***alloc_complex_tensor(long Nx, long Ny, long Nz) {
   long cnti, cntj;
   double complex ***tensor;
   
   if((tensor = (double complex ***) malloc((size_t) (Nx * sizeof(double complex **)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0] = (double complex **) malloc((size_t) (Nx * Ny * sizeof(double complex *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0][0] = (double complex *) malloc((size_t) (Nx * Ny * Nz * sizeof(double complex)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   for(cntj = 1; cntj < Ny; cntj ++)
      tensor[0][cntj] = tensor[0][cntj-1] + Nz;
   for(cnti = 1; cnti < Nx; cnti ++) {
      tensor[cnti] = tensor[cnti - 1] + Ny;
      tensor[cnti][0] = tensor[cnti - 1][0] + Ny * Nz;
      for(cntj = 1; cntj < Ny; cntj ++)
         tensor[cnti][cntj] = tensor[cnti][cntj - 1] + Nz;
   }
   
   return tensor;
}

/**
 *    Free double vector
 */
void free_double_vector(double *vector) {
   free((char *) vector);
}

/**
 *    Free complex vector
 */
void free_complex_vector(double complex *vector) {
   free((char *) vector);
}

/**
 *    Free double matrix
 */
void free_double_matrix(double **matrix) {
   free((char *) matrix[0]);
   free((char *) matrix);
}

/**
 *    Free complex matrix
 */
void free_complex_matrix(double complex **matrix) {
   free((char *) matrix[0]);
   free((char *) matrix);
}

/**
 *    Free double tensor
 */
void free_double_tensor(double ***tensor) {
   free((char *) tensor[0][0]);
   free((char *) tensor[0]);
   free((char *) tensor);
}

/**
 *    Free complex tensor
 */
void free_complex_tensor(double complex ***tensor) {
   free((char *) tensor[0][0]);
   free((char *) tensor[0]);
   free((char *) tensor);
}
