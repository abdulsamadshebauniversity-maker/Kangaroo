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
* GPU/GPUBsgs.cpp - host driver of the GPU BSGS search.
*
* Under nvcc this file is compiled as C++ (Makefile: g++), the kernels live in
* GPU/GPUBsgsKernel.h and the only CUDA specific calls are the memory ones.
* Under -DKW_EMULATE the very same code runs against GPU/GPUCompat.h and the
* kernels are executed by the SIMT emulator, which is what the test suite uses.
*/

#include "GPUBsgs.h"
#include "GPUBsgsKernel.h"

#include "GPULaunch.h"          // kernel launches (nvcc or SIMT emulator)

#include <cstdio>
#include <cstring>
#include <algorithm>

#ifndef KW_EMULATE
#include <cuda_runtime.h>
#endif

#define KW_BSGS_GRID 512

GPUBSGS::GPUBSGS(Secp256K1 *_secp) {

  secp = _secp;
  startKey.SetInt32(0);
  tableSize = 0;
  nbBaby = 0;
  nbGiant = 0;
  nbBucket = 0;
  hasTarget = false;
  maxFound = 256;
  found = false;
  foundGiant = 0;
  foundBaby = 0;
  timerBaby = 0;
  timerGiant = 0;
  nbCandidateTotal = 0;

  devGT = NULL;
  devKangarooGiant = NULL;
  devBabyX = NULL;
  devBabyFp = NULL;
  devBucketStart = NULL;
  devBucketFp = NULL;
  devBucketPos = NULL;
  devKangaroo = NULL;
  devResults = NULL;

}

GPUBSGS::~GPUBSGS() {

  if (devGT) cudaFree(devGT);
  if (devKangarooGiant) cudaFree(devKangarooGiant);
  if (devBabyX) cudaFree(devBabyX);
  if (devBabyFp) cudaFree(devBabyFp);
  if (devBucketStart) cudaFree(devBucketStart);
  if (devBucketFp) cudaFree(devBucketFp);
  if (devBucketPos) cudaFree(devBucketPos);
  if (devKangaroo) cudaFree(devKangaroo);
  if (devResults) cudaFree(devResults);

}

uint32_t GPUBSGS::BucketOf(uint64_t fp, uint32_t nbBucket) {
  return (uint32_t)(kwMixStep(fp) % (uint64_t)nbBucket);
}

void GPUBSGS::SetParams(const Int &start, uint64_t m, uint64_t n) {

  startKey.Set((Int *)&start);
  nbBaby = (m / KW_GROUP) * KW_GROUP;
  if (nbBaby == 0)
    nbBaby = KW_GROUP;
  nbGiant = n;
  tableSize = 0;

  // One bucket per KW_BUCKET_SIZE babies (25 % load factor), rounded up to a
  // power of two so the buckets stay small and the device loop short.
  uint32_t want = (uint32_t)((nbBaby + KW_BUCKET_SIZE - 1) / KW_BUCKET_SIZE);
  uint32_t nb = 1;
  while (nb < want)
    nb <<= 1;
  nbBucket = nb;

  // Build gTable: t*G (1..KW_HALF) | KW_GROUP*G | m*G | -(KW_HALF*G) | -G
  std::vector<uint64_t> table(2 * KW_HALF * 4 + 16, 0);
  for (int t = 1; t <= KW_HALF; t++) {
    Int k;
    k.SetInt32(t);
    Point P = secp->ComputePublicKey(&k);
    memcpy(&table[(t - 1) * 4], P.x.bits64, 32);
    memcpy(&table[(KW_HALF + t - 1) * 4], P.y.bits64, 32);
  }
  {
    Int k;
    k.SetInt32(KW_GROUP);
    Point minG = secp->ComputePublicKey(&k);
    memcpy(&table[2 * KW_HALF * 4], minG.x.bits64, 32);
    memcpy(&table[2 * KW_HALF * 4 + 4], minG.y.bits64, 32);
    // m*G : only x needed, the host negates y on the device side
    Int km;
    km.SetInt32((int64_t)nbBaby);
    Point mG = secp->ComputePublicKey(&km);
    memcpy(&table[2 * KW_HALF * 4 + 8], mG.x.bits64, 32);
    memcpy(&table[2 * KW_HALF * 4 + 12], mG.y.bits64, 32);
  }
  tableSize = table.size();

  if (devGT) cudaFree(devGT);
  cudaMalloc((void **)&devGT, table.size() * sizeof(uint64_t));
  cudaMemcpy(devGT, table.data(), table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

  if (devBabyX) cudaFree(devBabyX);
  cudaMalloc((void **)&devBabyX, (size_t)nbBaby * 4 * sizeof(uint64_t));
  if (devBabyFp) cudaFree(devBabyFp);
  cudaMalloc((void **)&devBabyFp, (size_t)nbBaby * sizeof(uint64_t));

  if (devBucketStart) cudaFree(devBucketStart);
  cudaMalloc((void **)&devBucketStart, ((size_t)nbBucket + 1) * sizeof(uint32_t));
  if (devBucketFp) cudaFree(devBucketFp);
  cudaMalloc((void **)&devBucketFp, (size_t)nbBaby * sizeof(uint64_t));
  if (devBucketPos) cudaFree(devBucketPos);
  cudaMalloc((void **)&devBucketPos, (size_t)nbBaby * sizeof(uint32_t));

  if (devResults) cudaFree(devResults);
  cudaMalloc((void **)&devResults, (1 + (size_t)maxFound * KW_GIANT_U32) * sizeof(uint32_t));

}

void GPUBSGS::SetTarget(Point *Q) {
  target = *Q;
  hasTarget = true;
}

// ---------------------------------------------------------------------------------
// Baby side: kangaroo centres for the walk, then the CSR bucket table.
// ---------------------------------------------------------------------------------

void GPUBSGS::BuildTable() {

  // number of threads used by the baby kernel
  uint64_t nbGroup = nbBaby / KW_GROUP;
  uint32_t nbThread = KW_BSGS_GRID;
  if (nbThread > nbGroup)
    nbThread = (uint32_t)nbGroup;
  uint64_t groupPerThread = (nbGroup + nbThread - 1) / nbThread;

  std::vector<uint64_t> kang((size_t)nbThread * 8, 0);
  for (uint32_t i = 0; i < nbThread; i++) {
    Int k;
    k.Set(&startKey);
    Int off;
    off.SetInt32(i * groupPerThread * KW_GROUP + KW_HALF);
    k.Add(&off);
    Point P = secp->ComputePublicKey(&k);
    memcpy(&kang[(size_t)i * 8], P.x.bits64, 32);
    memcpy(&kang[(size_t)i * 8 + 4], P.y.bits64, 32);
  }

  if (devKangaroo) cudaFree(devKangaroo);
  cudaMalloc((void **)&devKangaroo, kang.size() * sizeof(uint64_t));
  cudaMemcpy(devKangaroo, kang.data(), kang.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

  KWBabyParams bp;
  bp.rangeStart = startKey.bits64[0];
  bp.nbBaby = nbBaby;
  bp.groupPerThread = groupPerThread;

  double t0 = kwSeconds();
  kwLaunchBaby(1, (int)nbThread, devKangaroo, devGT, devBabyX, bp);
  cudaDeviceSynchronize();
  timerBaby = kwSeconds() - t0;

  // fingerprints of the babies (on the device)
  uint32_t nbBl = (uint32_t)((nbBaby + 255) / 256);
  kwLaunchFingerprint((int)nbBl, 256, devBabyX, devBabyFp, (uint32_t)nbBaby);
  cudaDeviceSynchronize();

  // counting sort on the host: exact bucket table, no capacity limit
  std::vector<uint64_t> fp(nbBaby);
  std::vector<uint64_t> bfpHost(nbBaby);
  cudaMemcpy(bfpHost.data(), devBabyFp, nbBaby * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  fp = bfpHost;

  std::vector<uint32_t> start(nbBucket + 1, 0);
  for (uint64_t i = 0; i < nbBaby; i++)
    start[BucketOf(fp[i], nbBucket) + 1]++;
  for (uint32_t b = 0; b < nbBucket; b++)
    start[b + 1] += start[b];

  std::vector<uint32_t> fill(start.begin(), start.end() - 1);
  std::vector<uint64_t> bFp(nbBaby, 0);
  std::vector<uint32_t> bPos(nbBaby, 0);
  for (uint64_t i = 0; i < nbBaby; i++) {
    uint32_t b = BucketOf(fp[i], nbBucket);
    uint32_t at = fill[b]++;
    bFp[at] = fp[i];
    bPos[at] = (uint32_t)i;
  }

  cudaMemcpy(devBucketStart, start.data(), (nbBucket + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devBucketFp, bFp.data(), nbBaby * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devBucketPos, bPos.data(), nbBaby * sizeof(uint32_t), cudaMemcpyHostToDevice);

}

// ---------------------------------------------------------------------------------
// Verification of a candidate: recompute the giant and the baby on the CPU.
// ---------------------------------------------------------------------------------

bool GPUBSGS::ScalarMultiplyGiant(uint64_t j, Point &out) {

  // B_j = Q - j*m*G
  Int big;
  big.SetInt32((int64_t)(j * nbBaby));
  if (big.IsZero()) {
    out = target;             // B_0 = Q
    return true;
  }
  Int neg;
  neg.Set(&secp->order);
  neg.Sub(&big);
  Point p = secp->ComputePublicKey(&neg);
  out = secp->AddDirect(target, p);
  return true;

}

bool GPUBSGS::ScalarMultiplyBaby(uint32_t i, Point &out) {

  Point p = secp->ComputePublicKey(&(const_cast<Int &>(startKey)));
  Int off;
  off.SetInt32((int64_t)i);
  Point pi = secp->ComputePublicKey(&off);
  out = secp->AddDirect(p, pi);
  return true;

}

void GPUBSGS::Verify(uint64_t giant, uint32_t babyPos, uint32_t) {

  if (found)
    return;

  nbCandidateTotal++;

  // 1. the candidate must be the baby of the same plain key
  Point b;
  ScalarMultiplyBaby(babyPos, b);

  // 2. full verification: Q == (start + j*m + i)*G
  Int key;
  key.Set(&startKey);
  Int off;
  off.SetInt32((int64_t)(giant * nbBaby + babyPos));
  key.Add(&off);
  Point check = secp->ComputePublicKey(&key);

  if (check.x.IsEqual(&target.x) && check.y.IsEqual(&target.y)) {
    found = true;
    foundGiant = giant;
    foundBaby = babyPos;
  }

}

// ---------------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------------

bool GPUBSGS::Search() {

  if (!hasTarget)
    return false;

  found = false;
  nbCandidateTotal = 0;

  BuildTable();

  // one thread per slice of giants, contiguous slices
  uint32_t nbThread = KW_BSGS_GRID;
  if ((uint64_t)nbThread > nbGiant)
    nbThread = (uint32_t)nbGiant;
  int64_t giantPerThread = (int64_t)((nbGiant + nbThread - 1) / nbThread);

  // kangaroos: B_j for the first giant of each thread, then -(m*G)
  std::vector<uint64_t> kang((size_t)nbThread * 8 + 8, 0);
  for (uint32_t i = 0; i < nbThread; i++) {
    Point B;
    ScalarMultiplyGiant((uint64_t)i * giantPerThread, B);
    memcpy(&kang[(size_t)i * 8], B.x.bits64, 32);
    memcpy(&kang[(size_t)i * 8 + 4], B.y.bits64, 32);
  }
  {
    Int mm;
    mm.SetInt32((int64_t)nbBaby);
    Point mG = secp->ComputePublicKey(&mm);
    memcpy(&kang[(size_t)nbThread * 8], mG.x.bits64, 32);
    memcpy(&kang[(size_t)nbThread * 8 + 4], mG.y.bits64, 32);
  }

  if (devKangarooGiant) cudaFree(devKangarooGiant);
  cudaMalloc((void **)&devKangarooGiant, kang.size() * sizeof(uint64_t));
  cudaMemcpy(devKangarooGiant, kang.data(), kang.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemset(devResults, 0, (1 + (size_t)maxFound * KW_GIANT_U32) * sizeof(uint32_t));

  KWGiantParams gp;
  gp.nbGiant = (int64_t)nbGiant;
  gp.giantPerThread = giantPerThread;
  gp.nbBucket = nbBucket;
  gp.maxFound = maxFound;

  double t0 = kwSeconds();
  kwLaunchGiant(1, (int)nbThread, devKangarooGiant, devBucketStart,
                devBucketFp, devBucketPos, devResults, gp);
  cudaDeviceSynchronize();
  timerGiant = kwSeconds() - t0;

  std::vector<uint32_t> res(1 + (size_t)maxFound * KW_GIANT_U32, 0);
  cudaMemcpy(res.data(), devResults, res.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  uint32_t n = res[0];
  if (n > maxFound) n = maxFound;
  for (uint32_t s = 0; s < n && !found; s++) {
    uint32_t *e = &res[1 + (size_t)s * KW_GIANT_U32];
    uint64_t giant = (uint64_t)e[0] | ((uint64_t)e[1] << 32);
    Verify(giant, e[2], e[3]);
  }

  return found;

}

Int GPUBSGS::GetKey() {

  Int key;
  key.Set(&startKey);
  if (found) {
    Int off;
    off.SetInt32((int64_t)(foundGiant * nbBaby + foundBaby));
    key.Add(&off);
  }
  return key;

}
