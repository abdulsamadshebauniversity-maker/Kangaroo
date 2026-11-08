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
* GPU/GPULaunch.h
*
* The only place where the kernels are launched. The host drivers
* (GPU/GPUBsgs.cpp, GPU/GPUXPoint.cpp) call these inline functions and never
* see a <<< >>>:
*
*   GPU build    : the drivers are compiled by nvcc (see the Makefile), KW_LAUNCH
*                  is the real kernel launch and the parameters travel by value.
*   -DKW_EMULATE : KW_LAUNCH runs the kernel through the SIMT emulator of
*                  GPUCompat.h, which is how the test suite exercises the very
*                  same sources on a machine without GPU or CUDA toolkit.
*
* Keeping the launch in one header means there is no second copy of the kernel
* calls or of the parameter marshalling that could drift away from the kernels.
* ---------------------------------------------------------------------------------
*/

#ifndef GPULAUNCH_H
#define GPULAUNCH_H

#include "GPUXPointKernel.h"
#include "GPUBsgsKernel.h"

// xpoint scan
inline void kwLaunchXPoint(int grid, int block,
                           uint64_t *kangaroos, const uint64_t *gTable,
                           const uint64_t *targets, uint32_t *results,
                           const KWScanParams &params, uint64_t *cursor) {

  KW_LAUNCH(kwXPointKernel, grid, block, kangaroos, gTable, targets, results,
            params, cursor);

}

// BSGS: baby table
inline void kwLaunchBaby(int grid, int block,
                         uint64_t *kangaroos, const uint64_t *gTable,
                         uint64_t *bX, const KWBabyParams &params) {

  KW_LAUNCH(kwBabyKernel, grid, block, kangaroos, gTable, bX, params);

}

// BSGS: baby fingerprints
inline void kwLaunchFingerprint(int grid, int block,
                                const uint64_t *bX, uint64_t *bFp,
                                uint32_t nbBaby) {

  KW_LAUNCH(kwFingerprintKernel, grid, block, bX, bFp, nbBaby);

}

// BSGS: giant steps
inline void kwLaunchGiant(int grid, int block,
                          uint64_t *kangaroos, const uint32_t *bucketStart,
                          const uint64_t *bucketFp, const uint32_t *bucketPos,
                          uint32_t *results, const KWGiantParams &params) {

  KW_LAUNCH(kwGiantKernel, grid, block, kangaroos, bucketStart, bucketFp,
            bucketPos, results, params);

}

#endif // GPULAUNCH_H
