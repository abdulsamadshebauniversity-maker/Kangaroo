/*
* This file is part of the Kangaroo distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

/*
* ---------------------------------------------------------------------------------
* GPU/GPUBsgsKernel.h
*
* Device side of the BSGS search: keyhunt MODE_BSGS ported to the grid.
*
* Interval to solve: [start, start + N) with N = m * nbGiant, target Q = K*G
*
*   baby  : b_i = (start + i)*G           i = 0 .. m-1
*   giant : B_j = Q - (j*m)*G             j = 0 .. nbGiant-1
*
*   B_j == b_i   <=>   K = start + j*m + i
*
* The two halves are split between the two kernels:
*
*  1. kwBabyKernel builds the m baby x coordinates with the affine group walk
*     (64 keys per iteration, one grouped inversion for the whole group), which
*     is the same cost model as the xpoint scan: ~1 inversion + 64 x-only
*     additions per 64 keys.
*
*  2. kwFingerprintKernel folds every baby x into a 64 bit fingerprint on the
*     device, and the host (GPUBsgs.cpp) builds a *counted* bucket table out of
*     them with a counting sort: bucket = mix(fp) % nbBucket, bucketStart[]
*     holds the prefix sum of the bucket sizes and the entries of bucket b are
*     bucketStart[b] .. bucketStart[b+1]-1. Nothing is ever dropped: the table
*     is exact whatever the bucket sizes are, the host only picks nbBucket so
*     that the average bucket holds KW_BUCKET_SIZE entries. Staying on the
*     device matters: a 2^24 baby table is 512 MB of x that never has to cross
*     PCIe just to be hashed.
*
*  3. kwGiantKernel walks the giants: every thread owns a contiguous slice of
*     the giant indices, keeps its base point B_j in registers and steps it by
*     -m*G, so a thread does nbGiant/nbThread complete additions and lookups,
*     one inversion each (kwAddPointAuto). Each giant is compared *once*: a
*     giant shifted by t*G would just be another giant index, so no group walk
*     is used here - the sqrt decomposition must stay exact.
*
* A 64 bit fingerprint hit is only a candidate; the host re-verifies every
* candidate with a full scalar multiplication (GPUBsgs.cpp), so a collision can
* never produce a wrong key, and since the fingerprints are exact for the m
* babies no key is ever missed.
* ---------------------------------------------------------------------------------
*/

#ifndef GPUBSGSKERNEL_H
#define GPUBSGSKERNEL_H

#include "GPUPointWalk.h"

#ifndef KW_BUCKET_SIZE
#define KW_BUCKET_SIZE 4
#endif

// ---------------------------------------------------------------------------------
// Complete addition with the inversion computed on the fly.
// Used only a couple of times per giant step (never inside the walk).
// ---------------------------------------------------------------------------------
KW_EMUL_STATIC __device__ void kwAddPointAuto(uint64_t *rx, uint64_t *ry,
                               uint64_t *px, uint64_t *py,
                               uint64_t *qx, uint64_t *qy) {

  uint64_t dx[4], inv[4];

  ModSub256(dx, qx, px);

  if ((dx[0] | dx[1] | dx[2] | dx[3]) == 0) {
    // doubling (P == Q) or point at infinity (P == -Q): kwAddPoint() handles
    // both, the inverse is unused
    kwAddPoint(rx, ry, px, py, qx, qy, dx);
    return;
  }

  kwInv(inv, dx);
  kwAddPoint(rx, ry, px, py, qx, qy, inv);

}

// ---------------------------------------------------------------------------------
// 1. Baby table
//
// The host gives each thread the centre of its first group (kangaroos) so the
// walk can start immediately. bX receives x((start + i)*G) for i in [0, m).
// ---------------------------------------------------------------------------------

struct KWBabyParams {
  uint64_t rangeStart;
  uint64_t nbBaby;          // m
  uint64_t groupPerThread;
};

KW_EMUL_STATIC __global__ void kwBabyKernel(uint64_t *kangaroos,
                             const uint64_t *gTable,
                             uint64_t *bX,
                             KWBabyParams p) {

  uint64_t cx[4], cy[4];
  uint64_t xs[KW_GROUP][4];

  const uint64_t *gx = gTable;
  const uint64_t *gy = gTable + KW_HALF * 4;
  const uint64_t *nx = gTable + 2 * KW_HALF * 4;
  const uint64_t *ny = gTable + 2 * KW_HALF * 4 + 4;

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  Load256(cx, kangaroos + (uint64_t)tid * 8);
  Load256(cy, kangaroos + (uint64_t)tid * 8 + 4);

  uint64_t g = (uint64_t)tid * p.groupPerThread;
  uint64_t gEnd = g + p.groupPerThread;
  if (gEnd > p.nbBaby / KW_GROUP)
    gEnd = p.nbBaby / KW_GROUP;

  for (; g < gEnd; g++) {

    kwWalkGroup(xs, cx, cy, cx, cy, gx, gy, nx, ny);

    uint64_t base = g * KW_GROUP;
    #pragma unroll 1
    for (int i = 0; i < KW_GROUP; i++)
      Load256(bX + (base + (uint64_t)i) * 4, xs[i]);

  }

}

// ---------------------------------------------------------------------------------
// 2. Baby fingerprints (device, so a large baby table never comes back to the
//    host just to be hashed)
// ---------------------------------------------------------------------------------

KW_EMUL_STATIC __global__ void kwFingerprintKernel(const uint64_t *bX, uint64_t *bFp,
                                    uint32_t nbBaby) {

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nbBaby)
    bFp[i] = kwFingerprint(bX + (uint64_t)i * 4);

}

// ---------------------------------------------------------------------------------
// 3. Giant steps
//
// The baby table is a CSR (counted) bucket table: nbBucket buckets,
// bucketStart[nbBucket+1] prefix sums, bucketFp/bucketPos the entries.
//
// Result record: giant index j (64 bit) + baby index i (32 bit) + the bucket
// that matched (32 bit, for the collision report of the host).
// ---------------------------------------------------------------------------------

#define KW_GIANT_U32 4

struct KWGiantParams {
  int64_t  nbGiant;
  int64_t  giantPerThread;
  uint32_t nbBucket;
  uint32_t maxFound;
};

KW_EMUL_STATIC __global__ void kwGiantKernel(uint64_t *kangaroos,
                              const uint32_t *bucketStart,
                              const uint64_t *bucketFp,
                              const uint32_t *bucketPos,
                              uint32_t *results,
                              KWGiantParams p) {

  uint64_t Bx[4], By[4];        // B_j = Q - j*m*G
  uint64_t mgx[4], mgy[4];      // -(m*G)

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // B_j for the first giant of this thread, supplied by the host
  Load256(Bx, kangaroos + (uint64_t)tid * 8);
  Load256(By, kangaroos + (uint64_t)tid * 8 + 4);

  // -(m*G) is uploaded right after the group table
  Load256(mgx, (uint64_t *)(kangaroos + (uint64_t)gridDim.x * blockDim.x * 8));
  Load256(mgy, (uint64_t *)(kangaroos + (uint64_t)gridDim.x * blockDim.x * 8 + 4));
  ModNeg256(mgy);                     // step is -m*G

  int64_t j = (int64_t)tid * p.giantPerThread;
  int64_t jEnd = j + p.giantPerThread;
  if (jEnd > p.nbGiant)
    jEnd = p.nbGiant;

  for (; j < jEnd; j++) {

    uint64_t fp = kwFingerprint(Bx);
    uint32_t b = (uint32_t)(kwMixStep(fp) % (uint64_t)p.nbBucket);
    uint32_t first = bucketStart[b];
    uint32_t end = bucketStart[b + 1];

    for (uint32_t s = first; s < end; s++) {
      if (bucketFp[s] == fp) {
        uint32_t slot = atomicAdd(results, 1u);
        if (slot < p.maxFound) {
          uint32_t *e = results + 1 + (uint64_t)slot * KW_GIANT_U32;
          e[0] = (uint32_t)((uint64_t)j & 0xFFFFFFFFULL);
          e[1] = (uint32_t)((uint64_t)j >> 32);
          e[2] = bucketPos[s];
          e[3] = b;
        }
        break;                       // one hit per giant is enough
      }
    }

    // B_{j+1} = B_j - m*G
    if (j + 1 < jEnd)
      kwAddPointAuto(Bx, By, Bx, By, mgx, mgy);

  }

}

#endif // GPUBSGSKERNEL_H
