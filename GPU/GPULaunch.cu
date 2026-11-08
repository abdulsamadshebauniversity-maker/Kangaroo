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
* GPU/GPULaunch.cu - real CUDA implementation of the launchers declared in
* GPU/GPULaunch.h. Only this file is compiled by nvcc; it includes the very
* same kernel headers the host drivers and the tests use.
*/

#include "GPULaunch.h"
#include "GPUXPointKernel.h"
#include "GPUBsgsKernel.h"

void kwLaunchXPoint(int grid, int block,
                    uint64_t *kangaroos, const uint64_t *gTable,
                    const uint64_t *targets, uint32_t *results,
                    const void *params, uint64_t *cursor) {

  KWScanParams p = *(const KWScanParams *)params;
  kwXPointKernel<<<grid, block>>>(kangaroos, gTable, targets, results, p, cursor);

}

void kwLaunchBaby(int grid, int block,
                  uint64_t *kangaroos, const uint64_t *gTable,
                  uint64_t *bX, const void *params) {

  KWBabyParams p = *(const KWBabyParams *)params;
  kwBabyKernel<<<grid, block>>>(kangaroos, gTable, bX, p);

}

void kwLaunchFingerprint(int grid, int block,
                         const uint64_t *bX, uint64_t *bFp, uint32_t nbBaby) {

  kwFingerprintKernel<<<grid, block>>>(bX, bFp, nbBaby);

}

void kwLaunchGiant(int grid, int block,
                   uint64_t *kangaroos, const uint32_t *bucketStart,
                   const uint64_t *bucketFp, const uint32_t *bucketPos,
                   uint32_t *results, const void *params) {

  KWGiantParams p = *(const KWGiantParams *)params;
  kwGiantKernel<<<grid, block>>>(kangaroos, bucketStart, bucketFp, bucketPos, results, p);

}
