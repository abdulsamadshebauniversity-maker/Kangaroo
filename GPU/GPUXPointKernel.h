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
* GPU/GPUXPointKernel.h
*
* Device side of the xpoint search (keyhunt MODE_XPOINT ported to CUDA).
*
* One CUDA thread owns one 64 key group. A thread repeatedly:
*   1. walks its group          -> 64 x coordinates      (kwWalkGroup)
*   2. hashes each x into a 64 bit fingerprint            (kwFingerprint)
*   3. compares with a small set of target fingerprints held in registers
*   4. advances the group centre and records the next starting key index
*
* The interval is split in one contiguous slice per thread. A thread starts at
* the first group of its slice (centre precomputed by the host with a single
* scalar multiplication) and then simply follows the walk chain, so the
* amortised cost per group is one grouped inversion plus one complete addition.
*
* Matches are written to a compact result buffer:
*    [0]                 : number of matches (atomic counter)
*    [1..]               : MATCH_SIZE uint32 per match:
*                          key offset (64 bit) + x (256 bit) + fingerprint
* The host only has to read back the counter (and the matches) so the PCIe
* traffic stays negligible even for huge scans.
* ---------------------------------------------------------------------------------
*/

#ifndef GPUXPOINTKERNEL_H
#define GPUXPOINTKERNEL_H

#include "GPUPointWalk.h"

// Maximum number of target fingerprints kept in registers per thread.
#ifndef KW_MAX_TARGETS
#define KW_MAX_TARGETS 32
#endif

// Result record: keyIndex(2 x 32) + x(8 x 32) + fingerprint(1 x 32) = 44 bytes
#define KW_MATCH_U32 11

struct KWScanParams {
  uint64_t rangeStart;       // key of xs[0] of group 0 (i.e. first key of the range)
  uint64_t nbGroup;          // number of groups of the whole range
  uint64_t groupPerThread;   // number of groups of each thread slice
  uint32_t nbTarget;         // number of target fingerprints
  uint32_t maxFound;         // result buffer capacity (in matches)
};

// ---------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------

KW_EMUL_STATIC __global__ void kwXPointKernel(uint64_t *kangaroos,
                               const uint64_t *gTable,
                               const uint64_t *targets,
                               uint32_t *results,
                               KWScanParams p,
                               uint64_t *cursor) {

  // Per thread: group centre, generated x, target fingerprints
  uint64_t cx[4], cy[4];
  uint64_t xs[KW_GROUP][4];
  uint64_t fp[KW_MAX_TARGETS];

  const uint64_t *gx = gTable;
  const uint64_t *gy = gTable + KW_HALF * 4;
  const uint64_t *nx = gTable + 2 * KW_HALF * 4;
  const uint64_t *ny = gTable + 2 * KW_HALF * 4 + 4;

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint32_t i = 0; i < p.nbTarget && i < KW_MAX_TARGETS; i++)
    fp[i] = targets[i];

  // Thread tid owns the contiguous slice [tid*groupPerThread,
  // (tid+1)*groupPerThread). The centre of its first group is precomputed by
  // the host, then the walk chain advances it by KW_GROUP keys at a time.
  Load256(cx, kangaroos + tid * 8);
  Load256(cy, kangaroos + tid * 8 + 4);

  uint64_t g = (uint64_t)tid * p.groupPerThread;
  uint64_t gEnd = g + p.groupPerThread;

  for (uint64_t nbDone = 0; g < gEnd; nbDone++) {

    uint64_t keyBase = p.rangeStart + g * KW_GROUP;     // key of xs[0]

    kwWalkGroup(xs, cx, cy, cx, cy, gx, gy, nx, ny);

    #pragma unroll 1
    for (int i = 0; i < KW_GROUP; i++) {
      uint64_t h = kwFingerprint(xs[i]);
      for (uint32_t t = 0; t < p.nbTarget; t++) {
        if (h == fp[t]) {
          uint32_t pos = atomicAdd(results, 1u);
          if (pos < p.maxFound) {
            uint32_t *e = results + 1 + pos * KW_MATCH_U32;
            uint64_t keyIdx = keyBase + (uint64_t)i;
            e[0] = (uint32_t)(keyIdx & 0xFFFFFFFFULL);
            e[1] = (uint32_t)(keyIdx >> 32);
            e[2] = (uint32_t)xs[i][0]; e[3] = (uint32_t)(xs[i][0] >> 32);
            e[4] = (uint32_t)xs[i][1]; e[5] = (uint32_t)(xs[i][1] >> 32);
            e[6] = (uint32_t)xs[i][2]; e[7] = (uint32_t)(xs[i][2] >> 32);
            e[8] = (uint32_t)xs[i][3]; e[9] = (uint32_t)(xs[i][3] >> 32);
            e[10] = (uint32_t)(h & 0xFFFFFFFFULL);
          }
          break;
        }
      }
    }

    g++;

  }

}

#endif // GPUXPOINTKERNEL_H
