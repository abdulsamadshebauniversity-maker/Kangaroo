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
* GPU/GPULaunchEmul.h
*
* -DKW_EMULATE implementation of the launchers of GPU/GPULaunch.h: the kernels
* are executed by the single threaded SIMT emulator of GPUCompat.h. Including
* this header in a driver gives a build that runs on a machine without GPU and
* without the CUDA toolchain, which is how the test suite validates the
* kernels; GPU/GPULaunch.cu provides the same entry points for nvcc.
*/

#ifndef GPULAUNCH_EMULATE_H
#define GPULAUNCH_EMULATE_H

#include "GPULaunch.h"
#include "GPUXPointKernel.h"
#include "GPUBsgsKernel.h"

inline void kwLaunchXPoint(int grid, int block,
                           uint64_t *kangaroos, const uint64_t *gTable,
                           const uint64_t *targets, uint32_t *results,
                           const void *params, uint64_t *cursor) {

  KWScanParams p = *(const KWScanParams *)params;
  KW_LAUNCH(kwXPointKernel, grid, block, kangaroos, gTable, targets, results, p, cursor);

}

inline void kwLaunchBaby(int grid, int block,
                         uint64_t *kangaroos, const uint64_t *gTable,
                         uint64_t *bX, const void *params) {

  KWBabyParams p = *(const KWBabyParams *)params;
  KW_LAUNCH(kwBabyKernel, grid, block, kangaroos, gTable, bX, p);

}

inline void kwLaunchFingerprint(int grid, int block,
                                const uint64_t *bX, uint64_t *bFp, uint32_t nbBaby) {

  KW_LAUNCH(kwFingerprintKernel, grid, block, bX, bFp, nbBaby);

}

inline void kwLaunchGiant(int grid, int block,
                          uint64_t *kangaroos, const uint32_t *bucketStart,
                          const uint64_t *bucketFp, const uint32_t *bucketPos,
                          uint32_t *results, const void *params) {

  KWGiantParams p = *(const KWGiantParams *)params;
  KW_LAUNCH(kwGiantKernel, grid, block, kangaroos, bucketStart, bucketFp, bucketPos, results, p);

}

#endif // GPULAUNCH_EMULATE_H
