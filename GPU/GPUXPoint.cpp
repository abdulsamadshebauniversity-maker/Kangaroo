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
* GPU/GPUXPoint.cpp - host driver of the GPU xpoint scan.
*/

#include "GPUXPoint.h"
#include "GPULaunch.h"

#include "../Timer.h"            // monotonic clock: kwSeconds() is in GPUPointWalk.h

#include <cstdio>
#include <cstring>

#ifndef KW_EMULATE
#include <cuda_runtime.h>
#endif

#define KW_XPOINT_GRID 512

GPUXPoint::GPUXPoint(Secp256K1 *_secp) {

  secp = _secp;
  startKey.SetInt32(0);
  nbGroup = 0;
  maxFound = 256;
  timerScan = 0;
  nbCandidateTotal = 0;

  devTable = NULL;
  devKangaroo = NULL;
  devTargets = NULL;
  devResults = NULL;

}

GPUXPoint::~GPUXPoint() {

  if (devTable) cudaFree(devTable);
  if (devKangaroo) cudaFree(devKangaroo);
  if (devTargets) cudaFree(devTargets);
  if (devResults) cudaFree(devResults);

}

void GPUXPoint::SetParams(const Int &start, uint64_t groupCount) {

  startKey.Set((Int *)&start);
  nbGroup = groupCount;
  foundKeys.clear();
  foundX.clear();

}

void GPUXPoint::AddTarget(Int *x) {

  if (targets.size() < KW_MAX_TARGETS)
    targets.push_back(*x);

}

void GPUXPoint::BuildTable() {

  // t*G for t = 1..KW_HALF, then KW_GROUP*G
  std::vector<uint64_t> table(2 * KW_HALF * 4 + 8, 0);
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
    Point nG = secp->ComputePublicKey(&k);
    memcpy(&table[2 * KW_HALF * 4], nG.x.bits64, 32);
    memcpy(&table[2 * KW_HALF * 4 + 4], nG.y.bits64, 32);
  }

  if (devTable) cudaFree(devTable);
  cudaMalloc((void **)&devTable, table.size() * sizeof(uint64_t));
  cudaMemcpy(devTable, table.data(), table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

}

void GPUXPoint::Verify(uint64_t key, uint64_t *x) {

  // 'key' is the absolute key, the way the kernel reports it
  nbCandidateTotal++;

  Int k;
  k.SetInt32((int64_t)key);

  Point P = secp->ComputePublicKey(&k);

  // the candidate must be a target (x match) - a 64 bit fingerprint collision
  // is rejected here
  for (size_t i = 0; i < targets.size(); i++) {
    if (targets[i].IsEqual(&P.x)) {
      Int px;
      px.SetInt32(0);
      memcpy(px.bits64, x, 32);
      if (px.IsEqual(&P.x)) {
        foundKeys.push_back(k);
        foundX.push_back(P.x);
      }
      return;
    }
  }

}

uint32_t GPUXPoint::Search() {

  foundKeys.clear();
  foundX.clear();
  nbCandidateTotal = 0;

  if (nbGroup == 0 || targets.empty())
    return 0;

  BuildTable();

  // one thread per slice of groups, contiguous slices
  uint32_t nbThread = KW_XPOINT_GRID;
  if ((uint64_t)nbThread > nbGroup)
    nbThread = (uint32_t)nbGroup;
  uint64_t groupPerThread = (nbGroup + nbThread - 1) / nbThread;

  // centre of the first group of each thread
  std::vector<uint64_t> kang((size_t)nbThread * 8, 0);
  for (uint32_t i = 0; i < nbThread; i++) {
    Int k;
    k.Set(&startKey);
    Int off;
    off.SetInt32((int64_t)((uint64_t)i * groupPerThread * KW_GROUP + KW_HALF));
    k.Add(&off);
    Point P = secp->ComputePublicKey(&k);
    memcpy(&kang[(size_t)i * 8], P.x.bits64, 32);
    memcpy(&kang[(size_t)i * 8 + 4], P.y.bits64, 32);
  }

  if (devKangaroo) cudaFree(devKangaroo);
  cudaMalloc((void **)&devKangaroo, kang.size() * sizeof(uint64_t));
  cudaMemcpy(devKangaroo, kang.data(), kang.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // target fingerprints, computed with the same mixer as the device
  std::vector<uint64_t> tf(targets.size(), 0);
  for (size_t i = 0; i < targets.size(); i++)
    tf[i] = kwFingerprint(targets[i].bits64);

  if (devTargets) cudaFree(devTargets);
  cudaMalloc((void **)&devTargets, tf.size() * sizeof(uint64_t));
  cudaMemcpy(devTargets, tf.data(), tf.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

  if (devResults) cudaFree(devResults);
  cudaMalloc((void **)&devResults, (1 + (size_t)maxFound * KW_MATCH_U32) * sizeof(uint32_t));
  cudaMemset(devResults, 0, (1 + (size_t)maxFound * KW_MATCH_U32) * sizeof(uint32_t));

  KWScanParams p;
  p.rangeStart = startKey.bits64[0];
  p.nbGroup = nbGroup;
  p.groupPerThread = groupPerThread;
  p.nbTarget = (uint32_t)tf.size();
  p.maxFound = maxFound;

  uint64_t cursor = 0;

  double t0 = kwSeconds();
  kwLaunchXPoint(1, (int)nbThread, devKangaroo, devTable, devTargets,
                 devResults, p, &cursor);
  cudaDeviceSynchronize();
  timerScan = kwSeconds() - t0;

  std::vector<uint32_t> res(1 + (size_t)maxFound * KW_MATCH_U32, 0);
  cudaMemcpy(res.data(), devResults, res.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  uint32_t n = res[0];
  if (n > maxFound) n = maxFound;
  for (uint32_t s = 0; s < n; s++) {
    uint32_t *e = &res[1 + (size_t)s * KW_MATCH_U32];
    uint64_t key = (uint64_t)e[0] | ((uint64_t)e[1] << 32);
    uint64_t x[4];
    x[0] = (uint64_t)e[2] | ((uint64_t)e[3] << 32);
    x[1] = (uint64_t)e[4] | ((uint64_t)e[5] << 32);
    x[2] = (uint64_t)e[6] | ((uint64_t)e[7] << 32);
    x[3] = (uint64_t)e[8] | ((uint64_t)e[9] << 32);
    Verify(key, x);
  }

  return (uint32_t)foundKeys.size();

}
