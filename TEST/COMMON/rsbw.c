/*******************************************************************************
 *   PRIMME PReconditioned Iterative MultiMethod Eigensolver
 *   Copyright (C) 2015 College of William & Mary,
 *   James R. McCombs, Eloy Romero Alcalde, Andreas Stathopoulos, Lingfei Wu
 *
 *   This file is part of PRIMME.
 *
 *   PRIMME is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU Lesser General Public
 *   License as published by the Free Software Foundation; either
 *   version 2.1 of the License, or (at your option) any later version.
 *
 *   PRIMME is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *   Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *******************************************************************************
 * File: rsbw.c
 * 
 * Purpose - librsb wrapper.
 * 
 ******************************************************************************/

#include <rsb.h>        /* for rsb_lib_init */
#include <blas_sparse.h>
#include <stdio.h>
#include <assert.h>
#include "primme.h"
#include "num.h"

int readMatrixRSB(const char* matrixFileName, blas_sparse_matrix *matrix, double *fnorm) {
#if !defined(RSB_NUMERICAL_TYPE_DOUBLE) || !defined(RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX)
   fprintf(stderr, "Needed librsb with support for 'double' and 'double complex'.\n");
   return -1;
#else
#  ifdef USE_DOUBLECOMPLEX
   rsb_type_t typecode = RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX;
#  else
   rsb_type_t typecode = RSB_NUMERICAL_TYPE_DOUBLE;
#  endif
   *matrix = blas_invalid_handle;
   if ((rsb_perror(NULL, rsb_lib_init(RSB_NULL_INIT_OPTIONS)))!=RSB_ERR_NO_ERROR) {
     fprintf(stderr, "Error while initializing librsb.\n");
     return -1;
   }

   *matrix = rsb_load_spblas_matrix_file_as_matrix_market(matrixFileName, typecode);
   if ( *matrix == blas_invalid_handle) {
      fprintf(stderr, "ERROR: Could not read matrix file\n");
      return -1;
   }

   if (!(BLAS_usgp(*matrix, blas_symmetric) == 1 || BLAS_usgp(*matrix, blas_hermitian) == 1)) {
      fprintf(stderr, "Matrix is not symmetric/Hermitian!"); 
      return -1;
   }
   assert(BLAS_ussp(*matrix, blas_rsb_autotune_next_operation) == 0);
   assert(BLAS_dusget_infinity_norm(*matrix, fnorm, blas_no_trans) == 0);

   return 0;
#endif
}

void RSBMatvec(void *x, void *y, int *blockSize, primme_params *primme) {
   int i;
   PRIMME_NUM *xvec, *yvec;
   blas_sparse_matrix *matrix;
#ifdef USE_DOUBLECOMPLEX
   const PRIMME_NUM one=(PRIMME_NUM)1;
#endif
   
   matrix = (blas_sparse_matrix *)primme->matrix;
   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;

   for (i=0; i<*blockSize*primme->nLocal; i++)
      yvec[i] = 0;
#ifndef USE_DOUBLECOMPLEX
   assert(BLAS_dusmm(blas_colmajor, blas_no_trans, *blockSize, 1.0, *matrix, xvec, primme->nLocal, yvec, primme->nLocal) == 0);
#else
   assert(BLAS_zusmm(blas_colmajor, blas_no_trans, *blockSize, &one, *matrix, xvec, primme->nLocal, yvec, primme->nLocal) == 0);
#endif
}
