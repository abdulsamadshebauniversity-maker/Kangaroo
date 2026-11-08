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
* GPUModes.cpp - command line front end of the two GPU search modes
*
*   ./kangaroo --mode xpoint --start <hex> --range <hex> --x <hex64> [--x <hex64>...]
*   ./kangaroo --mode bsgs   --start <hex> --range <hex> --pubkey <hex> [--m <baby>]
*
* The whole range is swept, all matches are printed one per line as
*
*   key=<hex> x=<hex64>
*
* Everything here is mode-agnostic glue: the search itself lives in
* GPU/GPUXPoint.cpp and GPU/GPUBsgs.cpp, which are compiled for both the
* emulator (-DKW_EMULATE) and a real CUDA device.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "SECPK1/SECP256k1.h"
#include "GPU/GPUXPoint.h"
#include "GPU/GPUBsgs.h"

static void usage() {

  printf("GPU search modes of Kangaroo\n\n");
  printf("  --mode xpoint --start <hex> --range <hex> --x <xhex> [--x <xhex> ...]\n");
  printf("      --start : first key of the interval (hex, default 0)\n");
  printf("      --range : number of keys to sweep (hex or decimal, default 1000)\n");
  printf("      --x     : x coordinate of a target point (64 hex chars), repeatable\n");
  printf("                (up to KW_MAX_TARGETS = 32 targets)\n\n");
  printf("  --mode bsgs --start <hex> --range <hex> --pubkey <hex> [--m <baby>]\n");
  printf("      --pubkey: public key of the target (130 hex chars, 04||x||y)\n");
  printf("      --m     : number of baby steps, multiple of 64 (default 65536)\n\n");
  printf("  --quiet : only print the matches\n");
  printf("  --help  : this message\n\n");
  printf("The interval of the xpoint mode is rounded up to a multiple of 64 keys.\n");
  printf("The BSGS interval is m*nbGiant: nbGiant is derived from --range.\n\n");

}

static Int parseInt(const char *s) {

  Int v;
  std::string str(s);
  if (str.size() > 2 && str[0] == '0' && (str[1] == 'x' || str[1] == 'X'))
    v.SetBase16((char *)str.substr(2).c_str());
  else if (str.size() > 32 || str.find_first_of("abcdefABCDEF") != std::string::npos)
    v.SetBase16((char *)str.c_str());        // long strings are hex
  else
    v.SetBase10((char *)str.c_str());
  return v;

}

static std::string toHex(Int v) {
  return v.GetBase16();
}

int gpuModeMain(int argc, char **argv, Secp256K1 *secp) {

  std::string mode;
  Int start;
  start.SetInt32(0);
  Int range;
  range.SetInt32(0x1000);
  std::vector<Int> targetsX;
  Point target;
  bool hasPub = false;
  bool quiet = false;
  uint64_t nbBaby = 65536;

  for (int a = 1; a < argc; a++) {

    std::string opt = argv[a];

    if (opt == "--mode" && a + 1 < argc) {
      mode = argv[++a];
    } else if (opt == "--start" && a + 1 < argc) {
      start = parseInt(argv[++a]);
    } else if (opt == "--range" && a + 1 < argc) {
      range = parseInt(argv[++a]);
    } else if (opt == "--x" && a + 1 < argc) {
      Int x;
      x.SetBase16(argv[++a]);
      targetsX.push_back(x);
    } else if (opt == "--pubkey" && a + 1 < argc) {
      bool compressed;
      if (!secp->ParsePublicKeyHex(std::string(argv[++a]), target, compressed)) {
        printf("Invalid public key\n");
        return 1;
      }
      hasPub = true;
    } else if (opt == "--m" && a + 1 < argc) {
      nbBaby = strtoull(argv[++a], NULL, 10);
    } else if (opt == "--quiet") {
      quiet = true;
    } else if (opt == "--help" || opt == "-h") {
      usage();
      return 0;
    } else {
      printf("Unknown option %s\n\n", argv[a]);
      usage();
      return 1;
    }

  }

  if (mode == "xpoint") {

    if (targetsX.empty()) {
      printf("xpoint mode needs at least one --x target\n");
      return 1;
    }

    uint64_t nbKey = (uint64_t)range.bits64[0];
    if (range.bits64[1] || range.bits64[2] || range.bits64[3]) {
      printf("xpoint mode: the range must fit in 64 bits\n");
      return 1;
    }
    uint64_t nbGroup = (nbKey + KW_GROUP - 1) / KW_GROUP;

    if (!quiet)
      printf("xpoint scan: %llu keys from %s (%llu groups, %u target(s))\n",
             (unsigned long long)(nbGroup * KW_GROUP), toHex(start).c_str(),
             (unsigned long long)nbGroup, (unsigned)targetsX.size());

    GPUXPoint xp(secp);
    xp.SetParams(start, nbGroup);
    for (size_t i = 0; i < targetsX.size(); i++)
      xp.AddTarget(&targetsX[i]);

    uint32_t n = xp.Search();
    for (uint32_t i = 0; i < n; i++)
      printf("key=%s x=%s\n", toHex(xp.GetFoundKey(i)).c_str(), toHex(xp.GetFoundX(i)).c_str());
    if (n == 0 && !quiet)
      printf("no match in the interval\n");
    return 0;

  }

  if (mode == "bsgs") {

    if (!hasPub) {
      printf("bsgs mode needs a --pubkey target\n");
      return 1;
    }

    uint64_t nbKey = (uint64_t)range.bits64[0];
    if (range.bits64[1] || range.bits64[2] || range.bits64[3] || nbKey == 0) {
      printf("bsgs mode: the range must fit in 64 bits and be non zero\n");
      return 1;
    }

    uint64_t m = (nbBaby / KW_GROUP) * KW_GROUP;
    if (m == 0) m = KW_GROUP;
    uint64_t n = (nbKey + m - 1) / m;

    if (!quiet)
      printf("bsgs search: %llu keys from %s (m=%llu babies, n=%llu giants)\n",
             (unsigned long long)(m * n), toHex(start).c_str(),
             (unsigned long long)m, (unsigned long long)n);

    GPUBSGS bsgs(secp);
    bsgs.SetParams(start, m, n);
    bsgs.SetTarget(&target);

    if (bsgs.Search()) {
      Int k = bsgs.GetKey();
      Point P = secp->ComputePublicKey(&k);
      printf("key=%s x=%s\n", toHex(k).c_str(), toHex(P.x).c_str());
    } else if (!quiet) {
      printf("no match in the interval\n");
    }
    return 0;

  }

  printf("Unknown or missing --mode (xpoint|bsgs)\n\n");
  usage();
  return 1;

}
