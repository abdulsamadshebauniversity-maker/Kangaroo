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
* GPU/GPUBsgs.h
*
* Host driver of the GPU BSGS search (GPU/GPUBsgsKernel.h).
*
*   GPUBSGS bsgs(secp);
*   bsgs.SetParams(startInt, nbBaby, nbGiant);   // m, n
*   bsgs.SetTarget(&Q);                          // Q = K*G
*   if (bsgs.Search()) {
*      // bsgs.GetKey() gives K (the interval offset relative to start)
*   }
*
* The class owns the device buffers (gTable, baby x, fingerprints, bucket
* table) and does the three verifications the kernel needs:
*   - the bucket table is built on the host with a counting sort, so a large
*     baby table is exact (no bucket capacity, nothing dropped),
*   - every candidate reported by the giant kernel is recomputed on the host
*     (Q - j*m*G == baby_i ?) before it is accepted,
*   - the reported key is re-derived from the interval start so the caller
*     never has to trust the arithmetic of the kernel.
* ---------------------------------------------------------------------------------
*/

#ifndef GPUBsgsH
#define GPUBsgsH

#include <vector>
#include "../SECPK1/SECP256k1.h"
#include "../Timer.h"
#include "GPUPointWalk.h"

class GPUBSGS {

public:

  GPUBSGS(Secp256K1 *secp);
  ~GPUBSGS();

  // m = number of baby steps (multiple of KW_GROUP), n = number of giant steps.
  // The interval covered is [0, m*n) keys starting at start.
  void SetParams(const Int &start, uint64_t nbBaby, uint64_t nbGiant);

  void SetTarget(Point *target);

  // Runs the search. Returns true when the key was found (GetKey() valid).
  bool Search();

  // Key K with K*G == target, relative to the interval start.
  Int GetKey();

  // Arithmetic work done by the last Search() (for the benchmarks/logs)
  void GetStats(double &secsBaby, double &secsGiant, uint64_t &nbCandidate) {
    secsBaby = timerBaby;
    secsGiant = timerGiant;
    nbCandidate = nbCandidateTotal;
  }

  // Fingerprint -> bucket used by the device; exposed for the tests so the
  // host and the kernel cannot drift apart.
  static uint32_t BucketOf(uint64_t fp, uint32_t nbBucket);

private:

  void BuildTable();
  void Verify(uint64_t giant, uint32_t babyPos, uint32_t bucket);
  bool ScalarMultiplyBaby(uint32_t babyPos, Point &out);
  bool ScalarMultiplyGiant(uint64_t giant, Point &out);

  Secp256K1 *secp;
  Int startKey;

  uint64_t nbBaby;
  uint64_t nbGiant;
  uint32_t nbBucket;
  size_t tableSize;

  bool hasTarget;
  Point target;

  // device buffers
  uint64_t *devGT;
  uint64_t *devKangarooGiant;
  uint64_t *devBabyX;
  uint64_t *devBabyFp;
  uint32_t *devBucketStart;
  uint64_t *devBucketFp;
  uint32_t *devBucketPos;
  uint64_t *devKangaroo;
  uint32_t *devResults;

  uint32_t maxFound;

  bool found;
  uint64_t foundGiant;
  uint32_t foundBaby;

  double timerBaby;
  double timerGiant;
  uint64_t nbCandidateTotal;

};

#endif // GPUBsgsH
