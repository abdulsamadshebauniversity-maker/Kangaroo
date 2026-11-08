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
* GPU/GPUXPoint.h
*
* Host driver of the GPU xpoint scan (GPU/GPUXPointKernel.h), the GPU version
* of keyhunt MODE_XPOINT.
*
*   GPUXPoint xp(secp);
*   xp.SetParams(startInt, nbGroup);      // interval = nbGroup * 64 keys
*   xp.AddTarget(&x);                     // 32 byte x coordinate
*   if (xp.Search()) { ... xp.GetFoundKey(i) ... }
*
* The class owns the device buffers (walk table, kangaroo centres, target
* fingerprints, result buffer) and re-verifies every match on the CPU before
* reporting it, exactly like the BSGS driver.
* ---------------------------------------------------------------------------------
*/

#ifndef GPUXPointH
#define GPUXPointH

#include <vector>
#include <string>
#include "../SECPK1/SECP256k1.h"
#include "GPUXPointKernel.h"

class GPUXPoint {

public:

  GPUXPoint(Secp256K1 *secp);
  ~GPUXPoint();

  // Interval [start, start + nbGroup*KW_GROUP)
  void SetParams(const Int &start, uint64_t nbGroup);
  void AddTarget(Int *x);
  uint32_t NbTarget() { return (uint32_t)targets.size(); }

  // One full sweep of the interval. Returns the number of verified matches.
  uint32_t Search();

  // Verified matches
  uint32_t NbFound() { return (uint32_t)foundKeys.size(); }
  Int GetFoundKey(uint32_t i) { return foundKeys[i]; }
  Int GetFoundX(uint32_t i) { return foundX[i]; }

private:

  void BuildTable();
  void Verify(uint64_t key, uint64_t *x);

  Secp256K1 *secp;
  Int startKey;
  uint64_t nbGroup;

  std::vector<Int> targets;
  std::vector<Int> foundKeys;
  std::vector<Int> foundX;

  uint64_t *devTable;
  uint64_t *devKangaroo;
  uint64_t *devTargets;
  uint32_t *devResults;

  uint32_t maxFound;
  double timerScan;
  uint64_t nbCandidateTotal;

};

#endif // GPUXPointH
