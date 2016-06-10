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
 * File: restart.c
 *
 * Purpose - Compute the Ritz vectors corresponding to the restartSize
 *           smallest eigenvalues.
 *
 ******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "primme.h"
#include "const.h"
#include "restart_d.h"
#include "restart_private_d.h"
#include "locking_d.h"
#include "ortho_d.h"
#include "solve_H_d.h"
#include "factorize_d.h"
#include "update_projection_d.h"
#include "update_W_d.h"
#include "convergence_d.h"
#include "numerical_d.h"


/*******************************************************************************
 * Subroutine: restart - This routine replaces V with V*c, some subset
 *             of the Ritz vectors of the current and the previous iteration.
 *             Related bases and matrices are updated accordingly.
 *
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 *
 * nLocal           Number of rows of V, W, Q, evecs and evecsHat assigned to the node
 *
 * ldV              The leading dimension of V, W, Q, evecs and evecsHat
 *
 * basisSize        The number of columns in V, W and Q
 *
 * resNorms         The residual norms of the converged eigenpairs
 *
 * previousHVecs    Coefficient vectors retained from the previous iteration
 *
 * ldpreviousHVecs  The leading dimension of previousHVecs
 *
 * numGuesses       Number of remaining initial guesses
 *
 * rwork            Real work array
 *
 * rworkSize        Must be of size 
 *
 * iwork            Integer work array
 *                  
 * primme           Structure containing various solver parameters
 *
 *
 * INPUT/OUTPUT ARRAYS AND PARAMETERS
 * ----------------------------------
 * restartSizeOutput The column of vectors of the output V.
 *
 * V                The orthonormal basis. After restart, contains Ritz vectors
 *                  plus the orthogonal components from numPrevRetained Ritz 
 *                  vectors from the penultimate step.
 *
 * W                A*V
 *
 * hU               The left singular vectors of R or the eigenvectors of QV/R
 *
 * ldhU             The leading dimension of the input hU
 *
 * newldhU          The leading dimension of the output hU
 *
 * hVecs            The eigenvectors of H
 *
 * ldhVecs          The leading dimension of the input hVecs
 *
 * newldhVecs       The leading dimension of the output hVecs
 *
 * hVals            The eigenvalues of H
 *
 * flags            Array indicating the convergence of the Ritz vectors
 *
 * iev              Array of size blockSize indicating which Ritz vectors are
 *                  targeted in the block
 *
 * ievSize          The length of iev
 *
 * blockNorms       The residual norms of the eigenpairs in the block
 *
 * evecs            The converged Ritz vectors. Without locking, all converged
 *                  eigenvectors are copied from V to evecs if skew projections
 *                  are required
 *
 * evals            The converged Ritz values
 *
 * resNorms         The residual norms of the converged eigenpairs
 *
 * evecsHat         K^{-1}evecs
 *
 * prevRitzVals     Projected values retained from the previous iteration
 *
 * numPrevRitzVals  Length of the vector prevRitzVals
 *
 * ldevecsHat       The leading dimension of evecsHat
 *
 * M, ldM           evecs'*evecsHat and the leading dimension of M
 *
 * UDU              The factorization of M
 *
 * ldUDU            The leading dimension of UDU
 *
 * ipivot           The pivot array of the UDU factorization
 *
 * H                The projection V'*A*V
 *
 * ldH              The leading dimension of H
 *
 * Q, R             The factors of the QR decomposition of (A - targetShift*B)*V
 *
 * ldQ, ldR         The leading dimension of Q and R
 *
 * numConverged     The number of converged eigenpairs
 *
 * numLocked        The number of locked eigenpairs
 *
 * numConvergedStored The # of converged vectors copied to evecs
 *
 * numPrevRetained  As input the number of columns of previousHVecs. As output the
 *                  number of columns added to V
 *
 * targetShiftIndex The target shift used in (A - targetShift*B) = Q*R
 *
 * restartsSinceReset Number of restarts since last reset of V and W
 *
 * reset            flag to reset V and W in the next restart
 *
 *
 * Return value
 * ------------
 * int   > 0 the restart size   
 *        -2 restart_H failed
 *        -4 factorization of M failed
 *        -5 flags do not correspond to converged pairs in pseudolocking
 *       
 ******************************************************************************/
 
int restart_dprimme(double *V, double *W, int nLocal, int basisSize, int ldV,
   double *hVals, double *hSVals, int *flags, int *iev, int *ievSize,
   double *blockNorms, double *evecs, int *evecsPerm, double *evals, double *resNorms,
   double *evecsHat, int ldevecsHat, double *M, int ldM, double *UDU,
   int ldUDU, int *ipivot, int *numConverged, int *numLocked,
   int *numConvergedStored, double *previousHVecs, int *numPrevRetained,
   int ldpreviousHVecs, int numGuesses, double *prevRitzVals, int *numPrevRitzVals,
   double *H, int ldH, double *Q, int ldQ, double *R, int ldR, double* QV, int ldQV,
   double *hU, int ldhU, int newldhU, double *hVecs, int ldhVecs, int newldhVecs,
   int *restartSizeOutput, int *targetShiftIndex, int *restartsSinceReset,
   int *reset, double machEps,
   double *rwork, int rworkSize, int *iwork, primme_params *primme) {

   int i;                   /* Loop indices */
   int restartSize;         /* Basis size after restarting                   */
   int indexOfPreviousVecs; /* Column index in hVecs with previous vecs      */
   int *iwork0;             /* Temporal integer workspace pointer            */
   int *restartPerm;        /* Permutation of hVecs used to restart V        */
   int *hVecsPerm;          /* Permutation of hVecs to sort as primme.target */
   int ret;                 /* returned error code                           */
   double aNorm = max((primme?primme->aNorm:0.0), primme->stats.estimateLargestSVal);

   /* Return memory requirement */

   if (V == NULL) {
      if (primme->locking) {
         rworkSize = restart_locking_dprimme(&basisSize, NULL,
               NULL, nLocal, NULL, 0, NULL, 0, basisSize, 0, NULL, NULL,
               NULL, 0, NULL, NULL, NULL, NULL, ievSize, NULL, NULL,
               NULL, numConverged, numConverged, NULL, NULL, 0, NULL,
               numPrevRetained, 0, NULL, NULL, 0, 0.0, NULL, 0,
               NULL, primme);
      }
      else {
         rworkSize = restart_soft_locking_dprimme(&basisSize, NULL, NULL,
               nLocal, NULL, 0, NULL, 0, basisSize, 0, NULL, NULL, NULL, 0,
               NULL, NULL, NULL, NULL, ievSize, NULL, NULL, NULL, NULL,
               evecsHat, 0, NULL, 0, numConverged, NULL, NULL,
               numPrevRetained, 0, NULL, NULL, 0, 0.0, NULL, 0,
               NULL, primme);

      }

      rworkSize += restart_projection_dprimme(NULL, 0, NULL, 0, NULL, 0, NULL, 0, 0,
            NULL, 0, NULL, 0, NULL, 0, 0,
            NULL, 0, 0, NULL, NULL, NULL, NULL, basisSize, basisSize,
            *numPrevRetained, basisSize, NULL,
            NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, 0, 0, NULL, NULL, 0.0, primme);

      return rworkSize;
   }

   /* ----------------------------------------------------------- */
   /* Special case: If (basisSize+numLocked) is the entire space, */
   /* then everything should be converged. Do not test, just flag */
   /* everything as converged                                     */
   /* ----------------------------------------------------------- */

   if (basisSize + *numLocked + primme->numOrthoConst >= primme->n) {
      for (i = 0; i < basisSize && *numConverged < primme->numEvals; i++)
         if (flags[i] == UNCONVERGED) { flags[i] = CONVERGED; (*numConverged)++; }
      restartSize = basisSize;
      *numPrevRetained = 0;
   }

   /* --------------------------------------------------------------------- */
   /* If basis isn't full, restart with the current basis size.             */
   /* If dynamic thick restarting is to be used, then determine the minimum */
   /* number of free spaces to be maintained and call the DTR routine.      */
   /* The DTR routine will determine how many coefficient vectors from the  */
   /* left and right of H-spectrum to retain at restart. If DTR is not used */
   /* then set the restart size to the minimum restart size.                */
   /* --------------------------------------------------------------------- */

   else if (basisSize <= primme->maxBasisSize - primme->maxBlockSize) {
       restartSize = basisSize;
   }
   else if (primme->restartingParams.scheme == primme_dtr) {
      int numFree = *numPrevRetained+max(3, primme->maxBlockSize);
      restartSize = dtr_dprimme(*numLocked, hVecs, hVals, flags, basisSize, numFree, 
            iev, rwork, primme);
   }
   else {
      restartSize = min(basisSize, primme->minRestartSize);
   }

   /* Update the estimation of the error in W and V */
   if (!*reset) {
      ++*restartsSinceReset;
   }
   else {
      *restartsSinceReset = 0;
   }
   primme->stats.estimateResidualError = 2*sqrt((double)*restartsSinceReset)*machEps*aNorm;
   
   restartPerm = iwork;
   hVecsPerm = &restartPerm[basisSize];
   iwork0 = &hVecsPerm[basisSize];

   if (!primme->locking) {
      double *X, *Res;
      ret = restart_soft_locking_dprimme(&restartSize, V, W, nLocal, R, ldR,
            hU, ldhU, basisSize, ldV, &X, &Res, hVecs, ldhVecs, restartPerm,
            hVals, flags, iev, ievSize, blockNorms, evecs, evals, resNorms, evecsHat,
            ldevecsHat, M, ldM, numConverged, numConvergedStored, previousHVecs,
            numPrevRetained, ldpreviousHVecs, &indexOfPreviousVecs, hVecsPerm, *reset, machEps,
            rwork, rworkSize, iwork0, primme);
   }
   else {
      double *X, *Res;
      ret = restart_locking_dprimme(&restartSize, V, W, nLocal, R,
            ldR, hU, ldhU, basisSize, ldV, &X, &Res, hVecs, ldhVecs,
            restartPerm, hVals, flags, iev, ievSize, blockNorms, evecs, evals, numConverged,
            numLocked, resNorms, evecsPerm, numGuesses, previousHVecs, numPrevRetained,
            ldpreviousHVecs, &indexOfPreviousVecs, hVecsPerm, *reset, machEps, rwork, rworkSize,
            iwork0, primme);
   }

   if (ret != 0) return ret;
   *reset = 0;

   /* Rearrange prevRitzVals according to restartPerm */

   if (primme->target != primme_smallest && primme->target != primme_largest) {
      permute_vecs_dprimme(prevRitzVals, 1, basisSize, 1, restartPerm, (double*)rwork, iwork0);
      permute_vecs_dprimme(prevRitzVals, 1, restartSize, 1, hVecsPerm, (double*)rwork, iwork0);
      *numPrevRitzVals = restartSize;
   }

   if (newldhVecs == 0) newldhVecs = restartSize;
   if (newldhU == 0) newldhU = restartSize;
   restart_projection_dprimme(V, ldV, W, ldV, H, ldH, Q, ldV, nLocal, R, ldR,
         QV, ldQV, hU, ldhU, newldhU, hVecs, ldhVecs, newldhVecs, hVals, hSVals,
         restartPerm, hVecsPerm, restartSize, basisSize, *numPrevRetained,
         indexOfPreviousVecs, evecs, numConvergedStored, primme->nLocal, evecsHat,
         ldevecsHat, M, ldM, UDU, ldUDU, ipivot, targetShiftIndex, *numConverged,
         rworkSize, rwork, iwork0, machEps, primme);

   *restartSizeOutput = restartSize; 

   return 0;
}

/*******************************************************************************
 * Subroutine: restart_soft_locking - This routine replaces V with V*c, some subset
 *             of the Ritz vectors, corresponding to the restartSize chosen
 *             eigenvalues of V'*A*V. It may include components from the 
 *             Ritz vectors from the (maxBasisSize-1) step (i.e., recurrence
 *             restarting).
 *
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 *
 * nLocal           Number of rows of V assigned to the node
 *
 * ldV              The leading dimension of V and W
 *
 * hR               The factors of the QR decomposition of (A - targetShift*B)*V
 *
 * ldhR             The leading dimension of Q and R
 *
 * hU               The eigenvectors of QV/R
 *
 * ldhU             The leading dimension of hU
 *
 * basisSize        Size of the basis V
 *
 * resNorms         The residual norms of the converged eigenpairs
 *
 * previousHVecs    Coefficient vectors retained from the previous iteration
 *
 * rwork            Real work array
 *
 * rworkSize        Must be of size 
 *
 * iwork            Integer work array
 *                  
 * primme           Structure containing various solver parameters
 *
 *
 * INPUT/OUTPUT ARRAYS AND PARAMETERS
 * ----------------------------------
 * restartSize      The number of vectors to restart with.
 *
 * V                The orthonormal basis. After restart, contains Ritz vectors
 *                  plus the orthogonal components from numPrevRetained Ritz 
 *                  vectors from the penultimate step.
 *
 * W                A*V
 *
 * X                Reference to the Ritz vectors of the eigenpairs in the block
 *
 * R                Reference to the residual vectors of the eigenpairs in the block
 *
 * hVecs, ldhVecs   The eigenvectors of H and the leading dimension of H
 *
 * restartPerm      The permutation applied to the columns of hVecs before restarting
 *
 * hVals            The eigenvalues of H
 *
 * flags            Array indicating the convergence of the Ritz vectors
 *
 * iev              Array of size blockSize indicating which Ritz vectors are
 *                  targeted in the block
 *
 * ievSize          The length of iev
 *
 * blockNorms       The residual norms of the eigenpairs in the block
 *
 * evecs            The converged Ritz vectors. Without locking, all converged
 *                  eigenvectors are copied from V to evecs if skew projections
 *                  are required
 *
 * evals            The converged Ritz values
 *
 * resNorms         The residual norms of the converged eigenpairs
 *
 * evecsHat         K^{-1}evecs
 *
 * ldevecsHat       The leading dimension of evecsHat
 *
 * M, ldM           evecs'*evecsHat and the leading dimension of M
 *
 * numConverged     The number of converged eigenpairs
 *
 * numConvergedStored The # of converged vectors copied to evecs
 *
 * numPrevRetained  As input the number of columns of previousHVecs. As output the
 *                  number of columns added to V
 *
 * indexOfPreviousVecs The first column in the output V that has a vector from previousHVecs
 *
 * hVecsPerm        The permutation that orders the output hVals and hVecs as primme.target
 *
 * 
 * OUTPUT ARRAYS AND PARAMETERS
 * ----------------------------
 * reset            flag to reset V and W in the next restart
 * 
 *
 * Return value
 * ------------
 * int   > 0 the restart size   
 *        -2 restart_H failed
 *        -4 factorization of M failed
 *        -5 flags do not correspond to converged pairs in pseudolocking
 *       
 ******************************************************************************/
 
static int restart_soft_locking_dprimme(int *restartSize, double *V, double *W, int nLocal,
   double *hR, int ldhR, double *hU, int ldhU,
   int basisSize, int ldV, double **X, double **R, double *hVecs, int ldhVecs,
   int *restartPerm, double *hVals, int *flags, int *iev, int *ievSize,
   double *blockNorms, double *evecs, double *evals, double *resNorms,
   double *evecsHat, int ldevecsHat, double *M, int ldM, int *numConverged,
   int *numConvergedStored, double *previousHVecs, int *numPrevRetained,
   int ldpreviousHVecs, int *indexOfPreviousVecs, int *hVecsPerm, int reset, double machEps,
   double *rwork, int rworkSize, int *iwork, primme_params *primme) {
 
   int i, j, ret;
   double tpone = +1.0e+00, tzero = +0.0e+00;             /*constants*/

   /* Return memory requirement */

   if (V == NULL) {
      double t;
      double d;
      return max(max(max(
                  nLocal,      /* permute_vecs for hVecs */
                  Num_reset_update_VWXR_dprimme(NULL, NULL, nLocal, basisSize, 0, &t,
                     *restartSize, 0, NULL,
                     &t, 0, *restartSize, 0,
                     &t, *numConverged, *numConverged+*ievSize, 0,
                     NULL, 0, 0, 0, 0,
                     &t, 0, *restartSize, 0,
                     &t, *numConverged, *numConverged+*ievSize, 0, &d,
                     NULL, 0, 0,
                     0, 0.0, NULL, 0, primme)),
                  /* if evecsHat, permutation matrix & compute_submatrix workspace */
                  evecsHat ? (primme->numOrthoConst+*numConverged)*
                     (primme->numOrthoConst+*numConverged)*2 : 0),
                  ortho_coefficient_vectors_dprimme(NULL, basisSize, 0, 0, *restartSize,
                     NULL, NULL, 0, NULL, 0, *numPrevRetained, 0.0, NULL, NULL, 0, primme));
   }

   /* -------------------------------------------------------------------------- */ 
   /* Check if any of the previous flagged converged eigenvalues seems           */
   /* to have become unconverged by checking hVals[i]-evals[i] < tol.            */
   /* If not, flag it UNCONVERGED and let it be targeted again. This avoids      */  
   /* early converged but unwanted evs preventing wanted from being targeted.    */
   /* -------------------------------------------------------------------------- */

   if (basisSize != primme->n) {
      for (i=0; i<primme->numEvals; i++) {
         if (flags[i] != UNCONVERGED && fabs(hVals[i]-evals[i]) > resNorms[i]) {
            flags[i] = UNCONVERGED;
         }
      }
   }

   /* ----------------------------------------------------------------- */
   /* Generate the permutation that put converged Ritz vectors first.   */
   /* Also update the number of converged values.                       */
   /* ----------------------------------------------------------------- */

   for (i=j=0, *numConverged=0; i<basisSize; i++)
      if (flags[i] != UNCONVERGED) restartPerm[j++] = i, (*numConverged)++;
   *numConverged = min(*numConverged, primme->numEvals);
   for (i=0; i<basisSize; i++)
      if (flags[i] == UNCONVERGED) restartPerm[j++] = i;

   permute_vecs_dprimme(hVecs, basisSize, basisSize, ldhVecs, restartPerm, rwork,
         iwork);

   /* Permute hVals */
   permute_vecs_dprimme(hVals, 1, basisSize, 1, restartPerm, (double*)rwork, iwork);

   /* ----------------------------------------------------------------------- */
   /* Restarting with a small number of coefficient vectors from the previous */
   /* iteration can be retained to accelerate convergence.  The previous      */
   /* coefficient vectors must be combined with the current coefficient       */
   /* vectors by first orthogonalizing the previous ones versus the current   */
   /* restartSize ones.  The orthogonalized previous vectors are then         */
   /* inserted into the hVecs array at hVecs(:,indexOfPreviousVecs).          */
   /* ----------------------------------------------------------------------- */

   *numPrevRetained = min(basisSize, *restartSize + *numPrevRetained)
      - *restartSize;
   *indexOfPreviousVecs = *restartSize;

   *restartSize += *numPrevRetained;

   Num_copy_matrix_dprimme(previousHVecs, basisSize, *numPrevRetained,
         ldpreviousHVecs, &hVecs[ldhVecs*(*indexOfPreviousVecs)], ldhVecs);

   ret = ortho_coefficient_vectors_dprimme(hVecs, basisSize, ldhVecs, *indexOfPreviousVecs,
         *restartSize, restartPerm, hU, ldhU, hR, ldhR, *numPrevRetained, machEps,
         iwork, rwork, rworkSize, primme);
   if (ret != 0) return ret;

   /* -------------------------------------------------------------- */
   /* Restart V and W by replacing it with the current Ritz vectors. */
   /* Compute X, R, blockNorms for the next values in the block.     */
   /* -------------------------------------------------------------- */

   if (X) {
      *ievSize = min(min(min(primme->maxBlockSize, primme->numEvals-*numConverged),
               *indexOfPreviousVecs-*numConverged), primme->maxBasisSize-*restartSize);
      *X = &V[*restartSize*ldV];
      *R = &W[*restartSize*ldV];
   }
   else {
      *ievSize = 0;
   }

   ret = Num_reset_update_VWXR_dprimme(V, W, nLocal, basisSize, ldV, hVecs,
         *restartSize, ldhVecs, hVals,
         V, 0, *restartSize, ldV,
         X?*X:NULL, *numConverged, *numConverged+*ievSize, ldV,
         NULL, 0, 0, 0, 0,
         W, 0, *restartSize, ldV,
         X?*R:NULL, *numConverged, *numConverged+*ievSize, ldV, blockNorms,
         NULL, 0, 0,
         reset, machEps, rwork, rworkSize, primme);
   if (ret != 0) return ret;

   if (X)
      for (i=0; i<*ievSize; i++)
         iev[i] = restartPerm[i+*numConverged];

   /* Undo the reordering of restartPerm */

   for (i=0; i<basisSize; i++)
      hVecsPerm[i] = restartPerm[i];
   permute_vecs_iprimme(hVecsPerm, basisSize, restartPerm, iwork);
   

   /* --------------------------------------------------------------------- */
   /* If the user requires (I-QQ') projectors in JDQMR without locking,     */
   /* the converged eigenvectors are copied temporarily to evecs. There     */
   /* they stay locked  for use in (I-QQ') and (I-K^{-1}Q () Q') projectors.*/
   /* NOTE THIS IS NOT LOCKING! The Ritz vectors remain in the basis, and   */
   /* they will overwrite evecs at the end.                                 */
   /* We recommend against this type of usage. It's better to use locking.  */
   /* --------------------------------------------------------------------- */

   /* Andreas NOTE: is done inefficiently for the moment. We should only */
   /* add the recently converged. But we need to differentiate them      */
   /* from flags...                                                      */

   if (evecsHat) {
      int newNumConvergedStored=0, oldSizeM, newSizeM;

      /* Pack evecs and evecsHat for the converged pairs restartPerm[0:numConverged] */

      for (i=0; i < *numConverged && restartPerm[i] < *numConvergedStored; i++) {
         Num_copy_matrix_dprimme(&evecs[(restartPerm[i]+primme->numOrthoConst)*nLocal],
               nLocal, 1, nLocal,
               &evecs[(newNumConvergedStored+primme->numOrthoConst)*nLocal],
               nLocal);
         Num_copy_matrix_dprimme(&evecsHat[(restartPerm[i]+primme->numOrthoConst)*ldevecsHat],
               nLocal, 1, ldevecsHat,
               &evecsHat[(newNumConvergedStored+primme->numOrthoConst)*ldevecsHat],
               ldevecsHat);
         newNumConvergedStored++;
      }

      /* Apply restartPerm to rows and columns of M */

      oldSizeM = *numConvergedStored + primme->numOrthoConst;
      newSizeM = newNumConvergedStored + primme->numOrthoConst;
      for (i=0; i<oldSizeM*newSizeM; i++)
         rwork[i] = tzero;
      for (i=0; i < primme->numOrthoConst; i++)
         rwork[oldSizeM*i + i] = tpone;
      for (; i < newSizeM; i++)
         rwork[oldSizeM*i + restartPerm[i]+primme->numOrthoConst] = tpone;
      compute_submatrix(rwork, newSizeM, oldSizeM, M, oldSizeM, ldM,
         M, ldM, rwork+oldSizeM*newSizeM, rworkSize-oldSizeM*newSizeM);

      *numConvergedStored = newNumConvergedStored;
   }

   return 0;
}

/******************************************************************************
 * Function Num_reset_update_VWXR - This subroutine performs the next operations:
 *
 *    X0 = V*h(nX0b+1:nX0e), X1 = V*h(nX1b+1:nX1e), evecs(evecsSize:) = V*h(nX2b+1:nX2e)
 *    if reset, then
 *       reortho evecs(evecsSize:)
 *       reortho X0
 *    end
 *    Wo = A*X0(nWob+1-nX0b:nWoe-nX0b),
 *    R = Wo(nRb+1-nWob:nRe-nWob) - X0(nRb+1-nX0b:nRe-nX0b)*diag(hVals(nRb+1:nRe)),
 *    Rnorms = norms(R),
 *    rnorms = norms(Wo(nrb+1-nWob:nre-nWob) - X0(nrb+1-nX0b:nre-nX0b)*diag(hVals(nrb+1:nre))),
 *
 * NOTE: if Rnorms and rnorms are requested, nRb-nRe+nrb-nre < mV
 *
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * V, W        input basis
 * mV,nV,ldV   number of rows and columns and leading dimension of V and W
 * h           input rotation matrix
 * nh          Number of columns of h
 * ldh         The leading dimension of h
 * hVals       Array of values
 *
 * OUTPUT ARRAYS AND PARAMETERS
 * ----------------------------
 * X0          Output matrix V*h(nX0b:nX0e-1) (optional)
 * nX0b, nX0e  Range of columns of h
 * X1          Output matrix V*h(nX1b:nX1e-1) (optional)
 * nX1b, nX1e  Range of columns of h
 * evecs       Output matrix V*h(nX2b:nX2e-1) (optional)
 * evecsSize   First column where to copy V*h(nX2b:nX2e-1) (optional)
 * nX2b, nX2e  Range of columns of h
 * Wo          Output matrix W*h(nWob:nWoe-1) (optional)
 * nWob, nWoe  Range of columns of h
 * R           Output matrix (optional)
 * nRb, nRe    Range of columns of h and hVals
 * Rnorms      Output array with the norms of R (optional)
 * rnorms      Output array with the extra residual vector norms (optional)
 * nrb, nre    Columns of residual vector to compute the norm
 * reset       if reset!=0, reothogonalize Xi and recompute Wo=A*X0
 * 
 * NOTE: n*e, n*b are zero-base indices of ranges where the first value is
 *       included and the last isn't.
 *
 ******************************************************************************/

int Num_reset_update_VWXR_dprimme(double *V, double *W, int mV, int nV, int ldV,
   double *h, int nh, int ldh, double *hVals,
   double *X0, int nX0b, int nX0e, int ldX0,
   double *X1, int nX1b, int nX1e, int ldX1,
   double *evecs, int evecsSize, int nX2b, int nX2e, int ldevecs,
   double *Wo, int nWob, int nWoe, int ldWo,
   double *R, int nRb, int nRe, int ldR, double *Rnorms,
   double *rnorms, int nrb, int nre,
   int reset, double machEps, double *rwork, int lrwork, primme_params *primme) {

   int i, j;         /* Loop variables */
   int m=min(PRIMME_BLOCK_SIZE, mV);   /* Number of rows in the cache */
   double *tmp, *tmp0;
   int ret;

   /* Return memory requirements and quick exit */
   if (V == NULL || reset == 0) {
      return Num_update_VWXR_dprimme(V, W, mV, nV, ldV, h, nh, ldh, hVals,
         X0, nX0b, nX0e, ldX0,
         X1, nX1b, nX1e, ldX1,
         evecs?&evecs[ldevecs*evecsSize]:NULL, nX2b, nX2e, ldevecs,
         Wo, nWob, nWoe, ldWo,
         R, nRb, nRe, ldR, Rnorms,
         rnorms, nrb, nre,
         rwork, lrwork, primme);
   }

   /* R or Rnorms or rnorms imply W */
   assert(!(R || Rnorms || rnorms) || W);

   assert(2*(nre-nrb) <= lrwork); /* Check workspace for tmp and tmp0 */

   /* X_i = V*h(nX_ib:nX_ie-1) */

   assert(!reset || !evecs || nX0b <= nX2b && nX2e <= nX0b);
   Num_update_VWXR_dprimme(V, NULL, mV, nV, ldV, h, nh, ldh, NULL,
         X0, nX0b, nX0e, ldX0,
         X1, nX1b, nX1e, ldX1,
         evecs?&evecs[ldevecs*evecsSize]:NULL, nX2b, nX2e, ldevecs,
         NULL, 0, 0, 0,
         NULL, 0, 0, 0, NULL,
         NULL, 0, 0,
         rwork, lrwork, primme);

   /* Reortho [X2 X0] against evecs if asked */

   ret = ortho_dprimme(evecs, ldevecs, NULL, 0, evecsSize, 
         evecsSize+nX2e-nX2b-1, NULL, 0, 0, mV, primme->iseed, 
         machEps, rwork, lrwork, primme);
   if (ret != 0) return ret;
   ret = ortho_dprimme(X0, ldX0, NULL, 0, 0, nX2b-nX0b-1, evecs,
         ldevecs, evecsSize+nX2e-nX2b, mV, primme->iseed, 
         machEps, rwork, lrwork, primme);
   if (ret != 0) return ret;
   Num_copy_matrix_dprimme(&evecs[ldevecs*evecsSize], mV, nX2e-nX2b,
         ldevecs, &X0[ldX0*(nX2b-nX0b)], ldX0);
   ret = ortho_dprimme(X0, ldX0, NULL, 0, nX2e-nX0b, nX0e-nX0b, evecs,
         ldevecs, evecsSize, mV, primme->iseed, machEps, rwork, lrwork,
         primme);
   if (ret != 0) return ret;

   /* Compute W = A*V for the orthogonalized corrections */

   assert(nWob == nX0b && nWoe == nX0e);
   matrixMatvec_dprimme(X0, mV, ldX0, Wo, ldWo, 0, nWoe-nWob, primme);
 
   /* R = Y(nRb-nYb:nRe-nYb-1) - X(nRb-nYb:nRe-nYb-1)*diag(nRb:nRe-1) */
   for (j=nRb; j<nRe; j++) {
      Num_compute_residual_dprimme(mV, hVals[j], &X0[ldX0*(j-nX0b)],
            &Wo[ldWo*(j-nWob)], &R[ldR*(j-nRb)]);
      if (Rnorms) {
         double ztmp;
         ztmp = Num_dot_dprimme(m, &R[i+ldR*(j-nRb)], 1, &R[i+ldR*(j-nRb)], 1);
         Rnorms[j-nRb] = *(double*)&ztmp;
      }
   }

   /* rnorms = Y(nrb-nYb:nre-nYb-1) - X(nrb-nYb:nre-nYb-1)*diag(nrb:nre-1) */
   if (rnorms) for (j=nrb; j<nre; j++) {
      rnorms[j-nrb] = 0.0;
      for (i=0; i < mV; i+=m, m=min(m,mV-i)) {
         double ztmp;
         Num_compute_residual_dprimme(m, hVals[j], &X0[ldX0*(j-nX0b)],
               &Wo[i+ldWo*(j-nWob)], rwork);
         ztmp = Num_dot_dprimme(m, rwork, 1, rwork, 1);
         rnorms[j-nrb] += *(double*)&ztmp;
      }
   }

   /* Reduce Rnorms and rnorms and sqrt the results */

   if (primme->globalSumDouble) {
      tmp = (double*)rwork;
      j = 0;
      if (Rnorms) for (i=nRb; i<nRe; i++) tmp[j++] = Rnorms[i-nRb];
      if (rnorms) for (i=nrb; i<nre; i++) tmp[j++] = rnorms[i-nrb];
      tmp0 = tmp+j;
      if (j) primme->globalSumDouble(tmp, tmp0, &j, primme);
      j = 0;
      if (Rnorms) for (i=nRb; i<nRe; i++) Rnorms[i-nRb] = sqrt(tmp0[j++]);
      if (rnorms) for (i=nrb; i<nre; i++) rnorms[i-nrb] = sqrt(tmp0[j++]);
   }
   else {
      if (Rnorms) for (i=nRb; i<nRe; i++) Rnorms[i-nRb] = sqrt(Rnorms[i-nRb]);
      if (rnorms) for (i=nrb; i<nre; i++) rnorms[i-nrb] = sqrt(rnorms[i-nrb]);
   }

   return 0; 
}


/*******************************************************************************
 * Subroutine: restart_projection - This routine updates Q, R, H and QV to match the
 *    changes in V.
 *
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * nLocal           Number of rows of V assigned to the node
 *
 * basisSize        Size of the basis V
 *
 * numConverged     The number of converged eigenpairs without locking
 *
 * numLocked        The number of Ritz vectors that have been locked 
 *
 * numPrevRetained  The number of coefficient vectors from previousHVecs in hVecs
 *  
 * indexOfPreviousVecs The first column in hVecs that has a vector from previousHVecs
 *
 * rwork            Real work array
 *
 * rworkSize        Size of rwork
 *                  
 * iwork            Integer work array
 *
 * primme           Structure containing various solver parameters
 *
 *
 * INPUT/OUTPUT ARRAYS AND PARAMETERS
 * ----------------------------------
 * restartSize      The number of vectors to restart with. If negative,
 *                  use dynamic thick restart.
 *
 * V                The orthonormal basis. After restart, contains Ritz vectors
 *                  plus the orthogonal components from numPrevRetained Ritz 
 *                  vectors from the penultimate step.
 *
 * ldV              The leading dimension of V
 *
 * W                A*V
 *
 * ldW              The leading dimension of W
 *
 * H                The projection V'*A*V
 *
 * ldH              The leading dimension of H
 *
 * Q, R             The factors of the QR decomposition of (A - targetShift*B)*V
 *
 * ldQ, ldR         The leading dimension of Q and R
 *
 * QV               = Q'*V
 *
 * ldQV             The leading dimension of QV
 *
 * hU               The left singular vectors of R or the eigenvectors of QV/R
 *
 * ldhU             The leading dimension of the input hU
 *
 * newldhU          The leading dimension of the output hU
 *
 * hVecs            The eigenvectors of H
 *
 * ldhVecs          The leading dimension of the input hVecs
 *
 * newldhVecs       The leading dimension of the output hVecs
 *
 * hVals            The eigenvalues of H
 *
 * hSVals           The singular values of R
 *
 * hVecsPerm        The permutation that orders hVals and hVecs as primme.target
 *
 * restartPerm      The permutation applied to the columns of hVecs before restarting
 *
 * flags            Array indicating the convergence of the Ritz vectors
 *
 * evecs            The converged Ritz vectors. Without locking, all converged
 *                  eigenvectors are copied from V to evecs if skew projections
 *                  are required.
 *
 * ldevecs          The leading dimension of evecs
 *
 * evecsHat         K^{-1}evecs
 *
 * ldevecsHat       The leading dimension of evecsHat
 *
 * M                evecs'*evecsHat
 *
 * ldM              The leading dimension of M
 *
 * UDU              The factorization of M
 *
 * ldUDU            The leading dimension of UDU
 *
 * ipivot           The pivot array of the UDU factorization
 *
 * targetShiftIndex The target shift used in (A - targetShift*B) = Q*R
 *
 *
 * Return value
 * ------------
 * int   > 0 the restart size   
 *        -2 restart_H failed
 *        -4 factorization of M failed
 *        -5 flags do not correspond to converged pairs in pseudolocking
 *       
 ******************************************************************************/
 
static int restart_projection_dprimme(double *V, int ldV, double *W, int ldW,
   double *H, int ldH, double *Q, int nLocal, int ldQ, double *R, int ldR,
   double *QV, int ldQV, double *hU, int ldhU, int newldhU, double *hVecs,
   int ldhVecs, int newldhVecs, double *hVals, double *hSVals, int *restartPerm,
   int *hVecsPerm, int restartSize, int basisSize, int numPrevRetained,
   int indexOfPreviousVecs, double *evecs, int *evecsSize,
   int ldevecs, double *evecsHat, int ldevecsHat, double *M, int ldM, double *UDU,
   int ldUDU, int *ipivot, int *targetShiftIndex, int numConverged,
   int rworkSize, double *rwork, int *iwork, double machEps, primme_params *primme) {

   int ret;

   /* -------------------------------------------------------- */
   /* Restart projected problem matrices H and R               */
   /* -------------------------------------------------------- */

   switch (primme->projectionParams.projection) {
   case primme_proj_RR:
      ret = restart_RR(H, ldH, hVecs, ldhVecs, newldhVecs, hVals, restartSize,
            basisSize, numConverged, numPrevRetained, indexOfPreviousVecs,
            hVecsPerm, rworkSize, rwork, iwork, primme);
      break;

   case primme_proj_harmonic:
      /* In harmonic extraction no vector comes from SVD, so they are treated in */
      /* the same way as the retained vectors from the previous iteration.       */

      indexOfPreviousVecs = 0;
      numPrevRetained = restartSize;

   case primme_proj_refined:
      ret = restart_qr(V, ldV, W, ldW, H, ldH, Q, nLocal, ldQ, R, ldR, QV, ldQV, hU, ldhU,
            newldhU, hVecs, ldhVecs, newldhVecs, hVals, hSVals, restartPerm, hVecsPerm,
            restartSize, basisSize, numPrevRetained, indexOfPreviousVecs, targetShiftIndex,
            numConverged, rworkSize, rwork, iwork, machEps, primme);
      break;

   default:
      assert(0);
   }

   if (H && ret != 0) {
      primme_PushErrorMessage(Primme_restart, Primme_restart_h, ret, __FILE__, 
         __LINE__, primme);
      return RESTART_H_FAILURE;
   }

   if (evecsHat) {
      int numRecentlyConverged = numConverged - *evecsSize;

      /* Return memory requirement */
      if (H == NULL) {
         return max(max(
               /* Workspace for restart_RR or restart_qr */
               ret,
               update_projection_dprimme(NULL, 0, NULL, 0, NULL, 0, nLocal,
                  *evecsSize, basisSize, NULL, 0, 1/*symmetric*/, primme)),
               UDUDecompose_dprimme(NULL, 0, NULL, 0, NULL, *evecsSize, NULL, 
                  0, primme));
      }

      /* Compute K^{-1}x for all newly locked eigenvectors */

      /* TODO: primme.shiftsForPreconditioner is undefined at that point;
         maybe it makes sense to always set NULL shiftsForPreconditioner
         when SkewQ is enabled to force the same preconditioner. */
      assert(ldevecs == primme->nLocal);
      primme->applyPreconditioner(&evecs[primme->nLocal*(*evecsSize+primme->numOrthoConst)],
            &evecsHat[primme->nLocal*(*evecsSize+primme->numOrthoConst)], &numRecentlyConverged,
            primme);
      primme->stats.numPreconds += numRecentlyConverged;

      /* Update the projection evecs'*evecsHat now that evecs and evecsHat   */
      /* have been expanded by numRecentlyConverged columns.  Required       */
      /* workspace is numLocked*numEvals.  The most ever needed would be     */
      /* maxBasisSize*numEvals.                                              */

      update_projection_dprimme(evecs, primme->nLocal, evecsHat, primme->nLocal,
            M, ldM, nLocal, *evecsSize+primme->numOrthoConst, numRecentlyConverged, rwork,
            rworkSize, 1/*symmetric*/, primme);
      *evecsSize = numConverged;

      ret = UDUDecompose_dprimme(M, ldM, UDU, ldUDU, ipivot, *evecsSize+primme->numOrthoConst,
            rwork, rworkSize, primme);

      if (ret != 0) {
         primme_PushErrorMessage(Primme_lock_vectors, Primme_ududecompose, ret,
            __FILE__, __LINE__, primme);
         return UDUDECOMPOSE_FAILURE;
      }

   }

   return 0;
}

/*******************************************************************************
 * Function restart_RR - This routine is used to recompute H = V'*A*V once V 
 *   has been restarted.  If no coefficient vectors from the previous iteration
 *   have been retained, then the restarted H will be diagonal and the 
 *   new eigenvectors (coefficient vectors) of H will be the standard basis
 *   vectors.  If previous coefficient vectors have been retained, then H will 
 *   contain the numPrevRetained x numPrevRetained submatrix 
 *   previousHVecs'*H*previousHvecs and the rest of the elements of H will be
 *   along the diagonal.
 *   
 *
 * INPUT PARAMETERS
 * ----------------
 * restartSize   Number of vectors the basis was restarted with
 * 
 * basisSize     Maximum size of the basis V
 *
 * numLocked     The number of Ritz vectors that have been locked 
 *
 * numPrevRetained The number of vectors retained from the previous iteration
 *
 * indexOfPreviousVecs  The index within hVecs where the previous vectors were
 *                      inserted.  Its also where the overlap matrix
 *                      previousHVecs'*H*previousHvecs will be inserted within
 *                      the restarted H.
 *
 * hVecsPerm     The permutation that orders hVals and hVecs as primme.target
 *
 * iwork         Integer work array
 *
 * rwork         Work array.  Necessary only when coefficient vectors from the
 *               previous iteration are retained.
 *
 * rworkSize     Size of rwork
 *
 *
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * H      Will contain H = V'*A*V given the restarted V.  H will be
 *        diagonal since V will contain only Ritz vectors, unless previous
 *        vectors are retained.  In that case, it will be diagonal except for
 *        a numPrevRetained x numPrevRetained submatrix inserted at
 *        H(numPrevRetained, numPrevRetained)
 *
 * ldH    The leading dimension of H
 *
 * hVecs  If the new H is diagonal, then it will contain the standard basis
 *        vectors.  If previous coefficient vectors are retained, then 
 *        restartSize - numPrevRetained of the vectors will be standard basis
 *        vectors.  The remaining numPrevRetained vectors will contain
 *        numPrevRetained non-zero elements corresponding to the 
 *        numPrevRetined x numPrevRetained submatrix.
 * 
 * ldhVecs  The leading dimension of the input hVecs
 *
 * newldhVecs  The leading dimension of the output hVecs
 *
 * hVals  The eigenvalues of the restarted H
 * 
 * 
 * Return value
 * ------------
 * Error code: 0 upon success
 *            -1 eigenvalues of submatrix could not be computed
 *
 ******************************************************************************/

static int restart_RR(double *H, int ldH, double *hVecs, int ldhVecs,
   int newldhVecs, double *hVals, int restartSize, int basisSize, int numLocked,
   int numPrevRetained, int indexOfPreviousVecs, int *hVecsPerm,
   int rworkSize, double *rwork, int *iwork, primme_params *primme) {

   int i, j;          /* Loop variables                                       */
   int ret;           /* Return value                                         */
   int orderedIndexOfPreviousVecs;  /* index of prev. vecs after applying hVecsPerm */
   double tpone = +1.0e+00, tzero = +0.0e+00;             /*constants*/

   /* Return memory requirement */

   if (H == NULL) {
      return max(
            compute_submatrix(NULL, numPrevRetained, 0, NULL, basisSize, 0, NULL,
               0, NULL, 0),
            solve_H_RR_dprimme(NULL, 0, NULL, 0, NULL, numPrevRetained,
               numLocked, 0, NULL, NULL, primme));
   }

   /* ---------------------------------------------------------------------- */
   /* If coefficient vectors from the previous iteration were retained, then */
   /* set up work space for computing the numPrevRetained x numPrevRetained  */
   /* submatrix, then compute the submatrix.                                 */
   /* ---------------------------------------------------------------------- */

   compute_submatrix(&hVecs[ldhVecs*indexOfPreviousVecs], numPrevRetained, ldhVecs, H,
         basisSize, ldH, &H[ldH*indexOfPreviousVecs+indexOfPreviousVecs],
         ldH, rwork, rworkSize);

   /* ----------------------------------------------------------------- */
   /* V*hVecs yields a diagonal matrix composed of the Ritz values of A */
   /* with respect to V.  Set H to a diagonal matrix with the Ritz      */
   /* values on the diagonal.                                           */
   /* ----------------------------------------------------------------- */

   for (j=0; j < indexOfPreviousVecs; j++) {
      for (i=0; i <= j; i++) {
         H[ldH*j+i] = tzero;
      }
      *(double*)&H[ldH*j+j] = hVals[j];
   }
   for (j=indexOfPreviousVecs; j<indexOfPreviousVecs+numPrevRetained; j++) {
      for (i=0; i < indexOfPreviousVecs; i++) {
         H[ldH*j+i] = tzero;
      }
   }
   for (j=indexOfPreviousVecs+numPrevRetained; j < restartSize; j++) {
      for (i=0; i <= j; i++) {
         H[ldH*j+i] = tzero;
      }
      *(double*)&H[ldH*j+j] = hVals[j];
   }

   /* --------------------------------------------------------------------- */
   /* Given the above H, we know the eigenvectors of H will be the standard */
   /* basis vectors if no previous coefficient vectors are retained         */
   /* --------------------------------------------------------------------- */

   for (j=0; j < restartSize; j++) {
      for (i=0; i < restartSize; i++) {
          hVecs[newldhVecs*j+i] = tzero;
      }
      hVecs[newldhVecs*j+hVecsPerm[j]] = tpone;
   }

   /* Apply permutation hVecsPerm to hVals */
   permute_vecs_dprimme(hVals, 1, restartSize, 1, hVecsPerm, (double*)rwork, iwork);

   /* ---------------------------------------------------------------------- */
   /* If coefficient vectors from the previous iteration have been retained, */
   /* then insert the computed overlap matrix into the restarted H and solve */ 
   /* the resulting eigenproblem for the resulting H.                        */
   /* ---------------------------------------------------------------------- */

   for (i=0, orderedIndexOfPreviousVecs=restartSize; i<restartSize; i++) {
      if (hVecsPerm[i] == indexOfPreviousVecs) {
         orderedIndexOfPreviousVecs = i;
         break;
      }
   }
   assert(orderedIndexOfPreviousVecs != restartSize || indexOfPreviousVecs >= restartSize);

   ret = solve_H_RR_dprimme(
         &H[ldH*indexOfPreviousVecs+indexOfPreviousVecs], ldH,
         &hVecs[newldhVecs*orderedIndexOfPreviousVecs+orderedIndexOfPreviousVecs],
         newldhVecs, &hVals[orderedIndexOfPreviousVecs], numPrevRetained,
         numLocked, rworkSize, rwork, iwork, primme);

   /* TODO: reorder all hVals */

   if (ret != 0) {
      primme_PushErrorMessage(Primme_restart_h, Primme_insert_submatrix, 
            ret, __FILE__, __LINE__, primme);
      return INSERT_SUBMATRIX_FAILURE;
   }

   return 0;
}

/*******************************************************************************
 * Function restart_qr - This routine is used to recompute the QR decomposition
 *    of W (=A*V), after V being replaced by V*hVecs. The coefficient vectors
 *    excepts hVecs(indexOfPrevRetained:indexOfPrevRetained+numPrevRetained) are
 *    right singular vectors of R. The output R for these columns will be
 *    a diagonal matrix with the singular values as diagonal elements. The
 *    output Q for these columns will be the update with the left singular
 *    vectors. The other columns of R and Q are recomputed properly without
 *    requiring explicitly recompute the QR factorization.
 *
 *    Also H = V'*A*V and QV = Q'*V are recomputed properly.
 *   
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * nLocal           Number of rows of V assigned to the node
 *
 * basisSize        Size of the basis V
 *
 * numConverged     The number of converged eigenpairs without locking
 *
 * numPrevRetained  The number of coefficient vectors from previousHVecs in hVecs
 *  
 * indexOfPreviousVecs The first column in hVecs that has a vector from previousHVecs
 *
 * rwork            Real work array
 *
 * rworkSize        Size of rwork
 *                  
 * iwork            Integer work array
 *
 * primme           Structure containing various solver parameters
 *
 *
 * INPUT/OUTPUT ARRAYS AND PARAMETERS
 * ----------------------------------
 * restartSize      The number of vectors to restart with. If negative,
 *                  use dynamic thick restart.
 *
 * V                The orthonormal basis. After restart, contains Ritz vectors
 *                  plus the orthogonal components from numPrevRetained Ritz 
 *                  vectors from the penultimate step.
 *
 * ldV              The leading dimension of V
 *
 * W                A*V
 *
 * ldW              The leading dimension of W
 *
 * H                The projection V'*A*V
 *
 * ldH              The leading dimension of H
 *
 * Q, R             The factors of the QR decomposition of (A - targetShift*B)*V
 *
 * ldQ, ldR         The leading dimension of Q and R
 *
 * QV               = Q'*V
 *
 * ldQV             The leading dimension of QV
 *
 * hU               The left singular vectors of R or the eigenvectors of QV/R
 *
 * ldhU             The leading dimension of the input hU
 *
 * newldhU          The leading dimension of the output hU
 *
 * hVecs            The eigenvectors of H
 *
 * ldhVecs          The leading dimension of the input hVecs
 *
 * newldhVecs       The leading dimension of the output hVecs
 *
 * hVals            The eigenvalues of H
 *
 * hSVals           The singular values of R
 *
 * hVecsPerm        The permutation that orders hVals and hVecs as primme.target
 *
 * restartPerm      The permutation applied to the columns of hVecs before restarting
 *
 * flags            Array indicating the convergence of the Ritz vectors
 *
 * targetShiftIndex The target shift used in (A - targetShift*B) = Q*R
 *
 * 
 * Return value
 * ------------
 * Error code: 0 upon success
 *            -1 eigenvalues of submatrix could not be computed
 *
 ******************************************************************************/

static int restart_qr(double *V, int ldV, double *W, int ldW, double *H,
   int ldH, double *Q, int nLocal, int ldQ, double *R, int ldR, double *QV,
   int ldQV, double *hU, int ldhU, int newldhU, double *hVecs, int ldhVecs,
   int newldhVecs, double *hVals, double *hSVals, int *restartPerm, int *hVecsPerm,
   int restartSize, int basisSize, int numPrevRetained, int indexOfPreviousVecs,
   int *targetShiftIndex, int numConverged, int rworkSize,
   double *rwork, int *iwork, double machEps, primme_params *primme) {

   int i, j;          /* Loop variables                                       */
   int ret;           /* Return value                                         */
   double tpone = +1.0e+00, tzero = +0.0e+00;             /*constants*/
   int orderedIndexOfPreviousVecs;  /* index of prev. vecs after applying hVecsPerm */

   /* Return memory requirement */
 
   if (H == NULL) {
      double t;
      int geqrfSize;    /* Workspace used by Num_geqrf_dprimme */
      int orgqrSize;    /* Workspace used by Num_orgqr_dprimme */

      Num_geqrf_dprimme(basisSize, numPrevRetained, NULL, basisSize, NULL, &t, -1, &ret);
      geqrfSize = *(double*)&t;
      Num_orgqr_dprimme(basisSize, numPrevRetained, numPrevRetained, NULL, basisSize, NULL,
         &t, -1, &ret);
      orgqrSize = *(double*)&t;

      return max(max(max(max(max(max(
         compute_submatrix(NULL, basisSize, 0, NULL, basisSize, 0, NULL, 0,
            NULL, 0),
         update_Q_dprimme(NULL, nLocal, 0, NULL, 0, NULL, 0, NULL, 0, 0.0, 0,
            basisSize, NULL, 0, 0.0, primme)),
         /* Workspace for  R(indexOfPrevVecs:) = R * hVecs(indexOfPrevVecs:) */
         basisSize*basisSize),
         /* Workspace for permute_vecs(hU) */
         basisSize),
         basisSize+max(geqrfSize, orgqrSize)),
         Num_update_VWXR_dprimme(NULL, NULL, nLocal, basisSize, 0, NULL, basisSize, 0, NULL,
            NULL, 0, 0, 0,
            NULL, 0, 0, 0,
            NULL, 0, 0, 0,
            NULL, 0, 0, 0,
            NULL, 0, 0, 0, NULL,
            NULL, 0, 0,
            NULL, 0, primme)),
         solve_H_dprimme(NULL, basisSize, 0, NULL, 0, NULL, 0, NULL, 0, NULL,
            0, NULL, NULL, numConverged, 0.0, 0, NULL, NULL, primme));
   }

   /* ------------------------------- */
   /* Replace H by hVecs' * H * hVecs */
   /* ------------------------------- */

   if (H) compute_submatrix(hVecs, restartSize, ldhVecs, H, basisSize, ldH, H, ldH,
      rwork, rworkSize);

   /* -------------------------------------- */
   /* Quick exit if the target has changed   */
   /* -------------------------------------- */

   if (*targetShiftIndex < 0 || primme->targetShifts[*targetShiftIndex]
         != primme->targetShifts[min(primme->numTargetShifts-1, numConverged)]) {

      *targetShiftIndex = min(primme->numTargetShifts-1, numConverged);

      ret = update_Q_dprimme(V, nLocal, ldV, W, ldW, Q, ldQ, R, ldR,
            primme->targetShifts[*targetShiftIndex], 0,
            restartSize, rwork, rworkSize, machEps, primme);
      if (ret != 0) return ret;

      if (QV) ret = update_projection_dprimme(Q, ldQ, V, ldV, QV, ldQV, nLocal, 0, restartSize,
            rwork, rworkSize, 0/*unsymmetric*/, primme);
      if (ret != 0) return ret;

      ret = solve_H_dprimme(H, restartSize, ldH, R, ldR, QV, ldQV, hU, newldhU, hVecs,
            newldhVecs, hVals, hSVals, numConverged, machEps, rworkSize, rwork, iwork, primme);
      if (ret != 0) return ret;

      return 0;

   }

   /* --------------- */
   /* QV = QV * hVecs */
   /* --------------- */
   
   if (QV) {
      Num_gemm_dprimme("N", "N", basisSize, restartSize, basisSize, tpone, QV, ldQV,
            hVecs, ldhVecs, tzero, rwork, basisSize);
      Num_copy_matrix_dprimme(rwork, basisSize, restartSize, basisSize, QV, ldQV);
   }

   /* -------------------------------------------------------------------- */
   /* During the restart V and W has replaced by V*hVecs and W*hVecs.      */
   /* Currently the QR decomposition correspond to W before restarting.    */
   /* To update the QR decomposition to the new W, replace Q by Q*Qn and   */
   /* R by Rn where Qn and Rn are the QR decomposition of R*hVecs = Qn*Rn. */
   /* -------------------------------------------------------------------- */

   /* -------------------------------------------------------------------- */
   /* R(indexOfPrevVecs:) = R * hVecs(indexOfPrevVecs:)                    */
   /* -------------------------------------------------------------------- */

   Num_copy_matrix_dprimme(&hVecs[indexOfPreviousVecs*ldhVecs], basisSize,
      numPrevRetained, ldhVecs, rwork, basisSize);
   Num_trmm_dprimme("L", "U", "N", "N", basisSize, numPrevRetained, tpone,
      R, ldR, rwork, basisSize);
   Num_copy_matrix_dprimme(rwork, basisSize, numPrevRetained, basisSize,
      &R[ldR*indexOfPreviousVecs], ldR);

   /* -------------------------------------------------------------------- */
   /* hVecs(0:indexOfPrevVecs) are the right singular vectors of R         */
   /* permuted with restartPerm. So                                        */
   /* R*hVecs(0:indexOfPrevVecs)=U(restartPerm)*diag(hSVals(restartPerm))  */
   /* -------------------------------------------------------------------- */

   for (j=0; j < indexOfPreviousVecs; j++) {
      for (i=0; i < primme->maxBasisSize; i++) {
         R[ldR*j+i] = tzero;
      }
      *(double*)&R[ldR*j+j] = hSVals[restartPerm[j]];
   }

   permute_vecs_dprimme(hU, basisSize, basisSize, ldhU, restartPerm, rwork, iwork);

   /* -------------------------------------------------------------------- */
   /* Compute the QR decomposition of R(indexOfPrevVecs:restartSize-1)     */
   /* -------------------------------------------------------------------- */

   Num_geqrf_dprimme(basisSize, numPrevRetained, &R[ldR*indexOfPreviousVecs],
      ldR, rwork, rwork+basisSize, rworkSize-basisSize, &ret);
   assert(ret == 0);

   /* -------------------------------------------------------------------- */
   /* Place the Q factor besides hU and apply permutation hVecsPerm        */
   /* -------------------------------------------------------------------- */

   Num_copy_matrix_dprimme(&R[ldR*indexOfPreviousVecs], basisSize,
      numPrevRetained, ldR, &hU[ldhU*indexOfPreviousVecs], ldhU);
   Num_orgqr_dprimme(basisSize, numPrevRetained, numPrevRetained,
      &hU[ldhU*indexOfPreviousVecs], ldhU, rwork, rwork+basisSize,
      rworkSize-basisSize, &ret);
   assert(ret == 0);

   /* -------------------------------------------------------------------- */
   /* Move the R factor to diagonal part of                                */
   /* R(indexOfPrevVecs:restartSize-1) and zero the lower triangular part  */
   /* of R(indexOfPrevVecs:restartSize-1, indexOfPrevVecs:restartSize-1).  */
   /* -------------------------------------------------------------------- */

   Num_copy_trimatrix_dprimme(&R[ldR*indexOfPreviousVecs], numPrevRetained,
      numPrevRetained, ldR, 0 /* copy upper part */, 0,
      &R[ldR*indexOfPreviousVecs+indexOfPreviousVecs],
      primme->maxBasisSize, 1 /* zero lower part */);

   /* -------------------------------------------------------------------- */
   /* Zero R(0:indexOfPrevVecs-1, indexOfPrevVecs:restartSize-1)           */
   /* -------------------------------------------------------------------- */

   for (j=indexOfPreviousVecs; j<indexOfPreviousVecs+numPrevRetained; j++) {
      for (i=0; i < indexOfPreviousVecs; i++) {
         R[ldR*j+i] = tzero;
      }
   }
   for (j=indexOfPreviousVecs+numPrevRetained; j < restartSize; j++) {
      for (i=0; i <= j; i++) {
         R[ldR*j+i] = tzero;
      }
      *(double*)&R[ldR*j+j] = hSVals[restartPerm[j]];
   }

   /* ----------------------------------- */
   /* Restart Q by replacing it with Q*hU */
   /* ----------------------------------- */

   Num_update_VWXR_dprimme(Q, NULL, nLocal, basisSize, ldQ, hU, restartSize,
      ldhU, NULL,
      Q, 0, restartSize, ldQ,
      NULL, 0, 0, 0,
      NULL, 0, 0, 0,
      NULL, 0, 0, 0,
      NULL, 0, 0, 0, NULL,
      NULL, 0, 0,
      rwork, rworkSize, primme);

   /* -------------- */
   /* QV = hU' * QV  */
   /* -------------- */
   
   if (QV) {
      Num_gemm_dprimme("C", "N", restartSize, restartSize, basisSize, tpone, hU, ldhU, QV, ldQV,
            tzero, rwork, restartSize);
      Num_copy_matrix_dprimme(rwork, restartSize, restartSize, restartSize, QV, ldQV);
   }

   /* ----------------------------------------------------------------------- */
   /* Given the above R, we know the right vectors will be the standard       */
   /* basis vectors if no previous coefficient vectors are retained           */
   /* ----------------------------------------------------------------------- */

   for (i=0, orderedIndexOfPreviousVecs=restartSize; i<restartSize; i++) {
      if (hVecsPerm[i] == indexOfPreviousVecs) {
         orderedIndexOfPreviousVecs = i;
         break;
      }
   }
   assert(orderedIndexOfPreviousVecs != restartSize || indexOfPreviousVecs >= restartSize);

   for (j=0; j < restartSize; j++) {
      for (i=0; i < restartSize; i++) {
          hVecs[newldhVecs*j+i] = tzero;
          hU[newldhU*j+i] = tzero;
      }
      if (j < orderedIndexOfPreviousVecs || j >= orderedIndexOfPreviousVecs+numPrevRetained) {
         hVecs[newldhVecs*j+hVecsPerm[j]] = tpone;
         hU[newldhU*j+hVecsPerm[j]] = tpone;
      }
   }

   if (!QV) {
      permute_vecs_dprimme(hSVals, 1, basisSize, 1, restartPerm, (double*)rwork, iwork);
      permute_vecs_dprimme(hSVals, 1, restartSize, 1, hVecsPerm, (double*)rwork, iwork);
      permute_vecs_dprimme(hVals, 1, restartSize, 1, hVecsPerm, (double*)rwork, iwork);
   }

   /* ---------------------------------------------------------------------- */
   /* If coefficient vectors from the previous iteration have been retained, */
   /* then insert the computed overlap matrix into the restarted R and solve */ 
   /* the resulting projected problem.                                       */
   /* ---------------------------------------------------------------------- */

   /* TODO: reorder all hVals when numPacked != 0 */

   if (QV) orderedIndexOfPreviousVecs = 0;
   ret = solve_H_dprimme(&H[ldH*indexOfPreviousVecs+indexOfPreviousVecs], numPrevRetained, ldH,
         &R[ldR*indexOfPreviousVecs+indexOfPreviousVecs], ldR, QV, ldQV,
         &hU[newldhU*orderedIndexOfPreviousVecs+orderedIndexOfPreviousVecs], newldhU,
         &hVecs[newldhVecs*orderedIndexOfPreviousVecs+orderedIndexOfPreviousVecs], newldhVecs,
         &hVals[orderedIndexOfPreviousVecs], &hSVals[orderedIndexOfPreviousVecs], numConverged,
         machEps, rworkSize, rwork, iwork, primme);
   if (ret != 0) {
      primme_PushErrorMessage(Primme_restart_h, Primme_insert_submatrix, 
            ret, __FILE__, __LINE__, primme);
      return INSERT_SUBMATRIX_FAILURE;
   }

   return 0;
}


/*******************************************************************************
 * Function dtr - This function determines the number of coefficient vectors
 *    to retain from both the left and right side of the spectrum.  The vectors
 *    are then copied so that they are contiguous in memory.
 *
 * Input parameters
 * ----------------
 * numLocked  The number of Ritz vectors that have been locked
 * 
 * basisSize  The current size of the basis
 *
 * numFree    Number of vacancies to be left in the basis
 *
 * iev        Array of size blockSize that determines index Ritz value index 
 *            each block vector is associated with.       
 *
 * rwork      double work array of size maxBasisSize^2. Use for hVec swapping.
 *
 * primme       Structure containing various solver parameters       
 *
 *
 * Input/output parameters
 * -----------------------
 * hVecs      The eigenvectors (coefficient vectors) of the projection V'*H*V
 *
 * hVals      The eigenvalues (Ritz values) of H
 *
 * flags      Array of size basisSize indicating the convergence of the Ritz
 *            vectors
 * 
 * Return value
 * ------------
 * int  The new restart size
 *
 ******************************************************************************/


static int dtr_dprimme(int numLocked, double *hVecs, double *hVals, int *flags, 
  int basisSize, int numFree, int *iev, double *rwork, primme_params *primme)
{

   int i;                 /* Loop variable */
   int l, lOpt, lMin;     /* Determine how many left side vectors to retain   */
   int r, rOpt;           /* Determine how many right side vectors to retain  */
   int maxIndex;          /* basisSize - 1                                    */
   int restartSize;       /* The new restart size                             */
   double currentRitzVal; /* The current Ritz value the solver is computing   */
   double newVal, optVal; /* Used to find the optimum gap ratio               */

   /* ---------------------------------------------------------------- */
   /* Compute lOpt and rOpt with respect to the first Ritz value being */
   /* targeted by the block.                                           */
   /* ---------------------------------------------------------------- */

   currentRitzVal = hVals[iev[0]];
   maxIndex = basisSize-1;

   /* If locking is engaged, then lMin must be large enough to retain */
   /* the coefficient vector associated with a converged target.      */
   /* lMin should be no smaller than primme->minRestartSize.            */

   if (primme->locking) {

      lMin = 0;

      /* Determine the largest index of any converged but unlocked target */
      /* Ritz vector.                                                     */

      for (l = 0; l < basisSize; l++) {
         if (flags[l] != UNCONVERGED && numLocked + l < primme->numEvals) {
            lMin = l;
         }
      }

      lMin = max(lMin, min(basisSize, primme->minRestartSize));

   }
   else {
      lMin = min(basisSize, primme->minRestartSize);
   }

   
   lOpt = lMin;
   rOpt = 0;   
   optVal = 0.0L;

   if (primme->printLevel >= 5 && primme->procID == 0) {
      fprintf(primme->outputFile,"DTR basisSize: %d\n", basisSize);
   }

   /* ---------------------------------------------------------------------- */
   /* Compute lOpt and rOpt that maximize the function.                      */
   /* maximize the function (basisSize-numFree-lMin-rMin)*                   */
   /*                       sqrt((currentRitzVal - hVals[l+1])/              */
   /*                            (hVals[l+1]-hVals[basisSize-1-r]))          */
   /* ---------------------------------------------------------------------- */

   for (l = lMin; l < basisSize - numFree; l++) {
      for (r = 0; r < basisSize - l - numFree; r++)       {
         if ((basisSize - l - r) % primme->maxBlockSize == 0) {
            newVal = (basisSize - l - r)
                     * sqrt((currentRitzVal - hVals[l+1])/
                            (hVals[l+1]-hVals[maxIndex-r]));

            if (newVal > optVal) {
               optVal = newVal;
               lOpt = l;
               rOpt = r;
            }

         }
      }
   }


   restartSize = lOpt + rOpt;

   /* --------------------------------------------------------------- */
   /* Swap the rOpt vectors from the right hand side so that they are */
   /* contiguous with the vectors from the left hand side.            */
   /* --------------------------------------------------------------- */

   i = basisSize - restartSize; 

   Num_dcopy_dprimme(i*basisSize, &hVecs[basisSize*lOpt], 1, rwork, 1);
   Num_dcopy_dprimme(rOpt*basisSize, &hVecs[basisSize*(basisSize-rOpt)], 1,
      &hVecs[basisSize*lOpt], 1);
   Num_dcopy_dprimme(i*basisSize, rwork, 1, &hVecs[basisSize*restartSize], 1);

   /* Do the same with the eigenvalues of H */

   Num_dcopy_primme(i, &hVals[lOpt], 1, (double *) rwork, 1);
   Num_dcopy_primme(rOpt, &hVals[(basisSize-rOpt)], 1, &hVals[lOpt], 1);
   Num_dcopy_primme(i, (double *) rwork, 1, &hVals[restartSize], 1);

   /* Set only those flags lower than restartSize. The rest will be reset */
   for (i = 0; i < rOpt; i++) {
      flags[lOpt + i] = flags[basisSize-rOpt + i];
   }

   if (primme->printLevel >= 5 && primme->procID == 0) {
      fprintf(primme->outputFile,"DTR restart size: %d L: %d R: %d\n", 
         restartSize, lOpt, rOpt);
   }

   reset_flags_dprimme(flags, restartSize, primme->maxBasisSize);
   return restartSize;

}

 
/*******************************************************************************
 * Subroutine compute_submatrix - This subroutine computes the nX x nX submatrix
 *    R = X'*H*X, where H stores the upper triangular part of a symmetric matrix.
 *    
 * Input parameters
 * ----------------
 * X        The coefficient vectors retained from the previous iteration
 *
 * nX       Number of columns of X
 *
 * H        Matrix
 *
 * nH       Dimension of H
 *
 * ldH      Leading dimension of H
 *
 * rwork    Work array.  Must be of size nH x nX
 *
 * lrwork   Length of the work array
 *
 * ldR      Leading dimension of R
 *
 * Output parameters
 * -----------------
 * R - nX x nX matrix computed 
 *
 ******************************************************************************/

static int compute_submatrix(double *X, int nX, int ldX, 
   double *H, int nH, int ldH, double *R, int ldR,
   double *rwork, int lrwork) {

   double tpone = +1.0e+00, tzero = +0.0e+00;

   /* Return memory requirement */
   if (X == NULL) {
      return nH*nX;
   }

   if (nH == 0 || nX == 0) return 0;

   assert(lrwork >= nH*nX);

   Num_symm_dprimme("L", "U", nH, nX, tpone, H, ldH, X, ldX, tzero, rwork, nH);
   
   Num_gemm_dprimme("C", "N", nX, nX, nH, tpone, X, ldX, rwork, nH, tzero, R, 
      ldR);

   return 0;
}


/*******************************************************************************
 * Subroutine reset_flags - Marks a series of Ritz values as unconverged
 *
 * INPUT PARAMETERS
 * ----------------
 * first Index of first Ritz value to be marked
 * last  Index of last Ritz value to be marked 
 *
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * flags The flags indicating which Ritz values have converged
 ******************************************************************************/
     
void reset_flags_dprimme(int *flags, int first, int last) {

   int i;  /* Loop variable */

   for (i = 0; i <= last; i++) {
      flags[i] = UNCONVERGED;
   }

}

/*******************************************************************************
 * Subroutine ortho_coefficient_vectors - Orthogonalize properly the columns of
 *    hVecs from indexOfPreviousVecs plus numPrevRetained.
 *    
 * INPUT/OUTPUT ARRAYS AND PARAMETERS
 * ----------------------------------
 * hVecs            The eigenvectors of H
 * ldhVecs          The leading dimension of the input hVecs
 * basisSize        Rows in hVecs and size of R and hU
 * newBasisSize     The number of columns in hVecs
 * perm             The permutation applied to the columns of hVecs before restarting
 * hU               The eigenvectors of QV/R
 * ldhU             The leading dimension of hU
 * R                The factors of the QR decomposition of (A - targetShift*B)*V
 * ldR              The leading dimension of R
 * iwork            Integer work array
 * rwork            Work array
 * rworkSize        Length of the work array
 *
 ******************************************************************************/

int ortho_coefficient_vectors_dprimme(double *hVecs, int basisSize, int ldhVecs,
   int indexOfPreviousVecs, int newBasisSize, int *perm, double *hU, int ldhU,
   double *R, int ldR, int numPrevRetained, int machEps, int *iwork,
   double *rwork, int rworkSize, primme_params *primme) {

   int ret;
   double tpone = +1.0e+00;

   if (hVecs && primme->projectionParams.projection == primme_proj_harmonic) {

      permute_vecs_dprimme(hU, basisSize, basisSize, ldhU, perm, rwork, iwork);
      Num_copy_matrix_dprimme(&hVecs[ldhVecs*indexOfPreviousVecs], basisSize, numPrevRetained,
            ldhVecs, &hU[ldhU*indexOfPreviousVecs], ldhU);
      ret = ortho_dprimme(hU, ldhU, NULL, 0, indexOfPreviousVecs,
            indexOfPreviousVecs+numPrevRetained-1,
            &hU[ldhU*(indexOfPreviousVecs+numPrevRetained)], ldhU,
            newBasisSize-indexOfPreviousVecs-numPrevRetained, basisSize,
            primme->iseed, machEps, rwork, rworkSize, NULL);
      if (ret != 0) return ret;
      Num_copy_matrix_dprimme(&hU[ldhU*indexOfPreviousVecs], basisSize, numPrevRetained,
            ldhU, &hVecs[ldhVecs*indexOfPreviousVecs], ldhVecs);
      Num_trsm_dprimme("L", "U", "N", "N", basisSize, numPrevRetained, tpone, R, ldR,
            &hVecs[ldhVecs*indexOfPreviousVecs], ldhVecs);

   }

   return ortho_dprimme(hVecs, ldhVecs, NULL, 0, indexOfPreviousVecs,
         indexOfPreviousVecs+numPrevRetained-1,
         &hVecs[ldhVecs*(indexOfPreviousVecs+numPrevRetained)], ldhVecs,
         newBasisSize-indexOfPreviousVecs-numPrevRetained, basisSize,
         primme->iseed, machEps, rwork, rworkSize, NULL);

}
