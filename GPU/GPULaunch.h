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
* The kernel <<< >>> launch syntax is only understood by nvcc, so the host
* drivers (GTUBsgs.cpp / GPUXPoint.cpp, compiled by g++) cannot call the
* kernels directly. This header is the thin boundary between the two:
*
*   KW_EMULATE  : the launchers are inline and run the kernels through the
*                 SIMT emulator of GPUCompat.h - one single build, no .cu file
*                 needed, which is what the test suite uses.
*   real CUDA   : the launchers are declared here and implemented in
*                 GPU/GPULaunch.cu (compiled by nvcc), which includes the very
*                 same kernel headers.
*
* In both cases the parameters are plain structures, so the ABI across the
* boundary is trivial.
* ---------------------------------------------------------------------------------
*/

#ifndef GPULAUNCH_H
#define GPULAUNCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct KWScanParams;
struct KWBabyParams;
struct KWGiantParams;

// xpoint scan
void kwLaunchXPoint(int grid, int block,
                    uint64_t *kangaroos, const uint64_t *gTable,
                    const uint64_t *targets, uint32_t *results,
                    const void *params, uint64_t *cursor);

// BSGS: baby table
void kwLaunchBaby(int grid, int block,
                  uint64_t *kangaroos, const uint64_t *gTable,
                  uint64_t *bX, const void *params);

// BSGS: baby fingerprints
void kwLaunchFingerprint(int grid, int block,
                         const uint64_t *bX, uint64_t *bFp, uint32_t nbBaby);

// BSGS: giant steps
void kwLaunchGiant(int grid, int block,
                   uint64_t *kangaroos, const uint32_t *bucketStart,
                   const uint64_t *bucketFp, const uint32_t *bucketPos,
                   uint32_t *results, const void *params);

#ifdef __cplusplus
}
#endif

#endif // GPULAUNCH_H
