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
* GPU/GPUPointWalk.h
*
* Affine group walk on SECPK1 for the GPU.
*
* Both the xpoint scan and BSGS need the x coordinates of a long run of
* consecutive points  P, P+G, P+2G, ...  The naive "one affine addition per key"
* costs one field inversion per key, which is far too slow. The trick used by
* the CPU implementation of keyhunt is kept: all the points of a group are
* derived from a single centre point, and P+tG / P-tG share the same
* dx = x(tG) - x(P), so a single grouped inversion covers the whole group.
*
* Layout of a group (KW_GROUP = 64 consecutive keys):
*
*   offset  -32 .. -1   ->  centre - tG     (t = 32 .. 1)
*   offset    0         ->  centre
*   offset  +1 .. +31   ->  centre + tG     (t = 1 .. 31)
*
* so a group covers centre + [-32,+31] and the next group starts at
* centre + 64G: consecutive groups cover the key space without hole nor overlap.
*
* Cost per group:
*   1_modinv of 33 values (one real inversion + 98 mults)
*   64 point additions (x only)  : 2 mults each
*   1 complete addition (+64G)   : 4 mults
*   => ~230 field mults for 64 keys, i.e. ~3.6 mult/key.
* ---------------------------------------------------------------------------------
*/

#ifndef GPUPOINTWALK_H
#define GPUPOINTWALK_H

#include "GPUMath.h"

// Number of keys per group and number of G multiples needed by kwWalkGroup().
// Inversion table layout: [0..KW_HALF-1] = G..KW_HALF*G , [KW_HALF] = KW_GROUP*G
#define KW_GROUP  64
#define KW_HALF   (KW_GROUP / 2)
#define KW_NDX    (KW_HALF + 1)

// ---------------------------------------------------------------------------------
// 64 bit fingerprint of a 256 bit value.
//
// xpoint search and BSGS both compare 32 byte x coordinates; instead of moving
// them back to the host they are reduced to 64 bits on the device with a fast
// mixer (identical on host and device, see tests). A 64 bit fingerprint is
// wide enough: a false positive needs a 64 bit collision and is always
// re-verified on the host with a full scalar multiplication.
// ---------------------------------------------------------------------------------

KW_HOST_DEVICE uint64_t kwMixStep(uint64_t x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27; x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  return x;
}

KW_HOST_DEVICE uint64_t kwFingerprint(const uint64_t *x) {
  uint64_t h = 0x9E3779B97F4A7C15ULL;
  h = kwMixStep(h ^ x[0]);
  h = kwMixStep(h ^ x[1]);
  h = kwMixStep(h ^ x[2]);
  h = kwMixStep(h ^ x[3]);
  return h;
}

// ---------------------------------------------------------------------------------
// Monotonic wall clock (seconds), used by the host drivers for the timings.
// ---------------------------------------------------------------------------------
#if defined(_WIN32)
#include <windows.h>
inline double kwSeconds() {
  LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
  return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#include <sys/time.h>
inline double kwSeconds() {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}
#endif

// ---------------------------------------------------------------------------------
// Field primitives
// ---------------------------------------------------------------------------------

// y^2 == x^3 + 7 (mod P)
KW_EMUL_STATIC __device__ bool kwOnCurve(uint64_t *x, uint64_t *y) {

  uint64_t x2[4], x3[4], y2[4], seven[4], t[4];

  _ModSqr(x2, x);
  _ModMult(x3, x2, x);
  _ModSqr(y2, y);
  seven[0] = 7; seven[1] = 0; seven[2] = 0; seven[3] = 0;
  ModSub256(t, x3, y2);
  ModSub256(t, t, seven);

  return (t[0] | t[1] | t[2] | t[3]) == 0;

}

// Modular inversion of a single 4 limb value.
// _ModInv() works on 5 limbs (320 bit signed), so a zeroed 5th limb is
// mandatory: passing a 4 limb array directly would read a garbage carry limb.
KW_EMUL_STATIC __device__ void kwInv(uint64_t r[4], uint64_t *a) {
  uint64_t t[5];
  Load256(t, a);
  t[4] = 0;
  _ModInv(t);
  Load256(r, t);
}

// Modular inversion of n values with a single inversion.
// v[i] is replaced by v[i]^-1 (mod P). n must be a compile time constant.
template <int N>
KW_EMUL_STATIC __device__ void kwGroupInv(uint64_t v[N][4]) {

  uint64_t sub[N][4];

  Load256(sub[0], v[0]);
  for (int i = 1; i < N; i++)
    _ModMult(sub[i], sub[i - 1], v[i]);

  uint64_t inv[5];
  Load256(inv, sub[N - 1]);
  inv[4] = 0;
  _ModInv(inv);

  uint64_t tmp[4];
  for (int i = N - 1; i > 0; i--) {
    _ModMult(tmp, sub[i - 1], inv);
    _ModMult(inv, inv, v[i]);
    Load256(v[i], tmp);
  }
  Load256(v[0], inv);

}

// rx = x(P + Q), given inv = (Q.x - P.x)^-1
KW_EMUL_STATIC __device__ void kwAddX(uint64_t *rx, uint64_t *px, uint64_t *py,
                       uint64_t *qx, uint64_t *qy, uint64_t *inv) {

  uint64_t dy[4], s[4], s2[4];

  ModSub256(dy, qy, py);
  _ModMult(s, dy, inv);
  _ModSqr(s2, s);
  ModSub256(rx, s2, px);
  ModSub256(rx, rx, qx);

}

// Complete affine addition ((rx,ry) = (px,py) + (qx,qy)).
// Returns false when the result is the point at infinity (P == -Q).
// Doubling (P == Q) is handled. Costs one inversion, only used once per
// group to move the centre by KW_GROUP*G.
KW_EMUL_STATIC __device__ bool kwAddPoint(uint64_t *rx, uint64_t *ry,
                           uint64_t *px, uint64_t *py,
                           uint64_t *qx, uint64_t *qy,
                           uint64_t *inv) {

  // rx/ry may alias px/py or qx/qy: all the results are accumulated in
  // temporaries and stored at the very end.
  uint64_t dx[4], dy[4], s[4], s2[4], t[4], z[4];
  uint64_t lx[4], ly[4];

  ModSub256(dx, qx, px);
  ModSub256(dy, qy, py);
  z[0] = dx[0] | dx[1] | dx[2] | dx[3];

  if (z[0] == 0) {
    z[0] = dy[0] | dy[1] | dy[2] | dy[3];
    if (z[0] != 0)
      return false;                       // P == -Q : point at infinity
    // Doubling: s = 3x^2 / 2y
    uint64_t x2[4], num[4], den[4], two[4], three[4];
    _ModSqr(x2, px);
    three[0] = 3; three[1] = 0; three[2] = 0; three[3] = 0;
    _ModMult(num, x2, three);
    two[0] = 2; two[1] = 0; two[2] = 0; two[3] = 0;
    _ModMult(den, py, two);
    kwInv(den, den);
    _ModMult(s, num, den);
  } else {
    _ModMult(s, dy, inv);
  }

  _ModSqr(s2, s);
  ModSub256(lx, s2, px);
  ModSub256(lx, lx, qx);                  // lx = s^2 - px - qx
  ModSub256(t, px, lx);                   // t  = px - lx
  _ModMult(ly, s, t);
  ModSub256(ly, ly, py);                  // ly = s(px - lx) - py

  Load256(rx, lx);
  Load256(ry, ly);

  return true;

}

// ---------------------------------------------------------------------------------
// Group walk
//
// gx[t-1], gy[t-1] : coordinates of t*G   for t in [1, KW_HALF]
// nx, ny           : coordinates of KW_GROUP*G
//
// In  : cx, cy       centre of the group
// Out : xs[KW_GROUP][4]  x coordinate of (cx,cy) + off*G with off = i - KW_HALF
//       ncx, ncy         new centre = old centre + KW_GROUP*G
//
// The x coordinates are produced in the "offset" order:
//   xs[i] = x(centre + (i - KW_HALF)*G)
// so xs[KW_HALF] is the centre itself.
// ---------------------------------------------------------------------------------
// sign = +1 : the walk enumerates centre + (i - KW_HALF)*G  and the centre
//             advances by +KW_GROUP*G
// sign = -1 : the walk enumerates centre - (i - KW_HALF)*G  and the centre
//             advances by -KW_GROUP*G (the giant steps of BSGS go backwards)
KW_EMUL_STATIC __device__ void kwWalkGroupSigned(uint64_t xs[KW_GROUP][4],
                                  uint64_t *ncx, uint64_t *ncy,
                                  uint64_t *cx, uint64_t *cy,
                                  const uint64_t *gx, const uint64_t *gy,
                                  const uint64_t *nx, const uint64_t *ny,
                                  int sign) {

  uint64_t dx[KW_NDX][4];
  uint64_t gxv[KW_HALF][4];
  uint64_t gyv[KW_HALF][4];

  // dx[t-1] = x(tG) - x(P) for the tG used on both sides
  for (int t = 1; t <= KW_HALF; t++) {
    Load256(gxv[t - 1], (uint64_t *)(gx + (t - 1) * 4));
    Load256(gyv[t - 1], (uint64_t *)(gy + (t - 1) * 4));
    if (sign < 0)
      ModNeg256(gyv[t - 1]);
    ModSub256(dx[t - 1], gxv[t - 1], cx);
  }
  // last entry: dx for the KW_GROUP*G step used for the next centre
  uint64_t nX[4], nY[4];
  {
    Load256(nX, (uint64_t *)nx);
    Load256(nY, (uint64_t *)ny);
    if (sign < 0)
      ModNeg256(nY);
    ModSub256(dx[KW_HALF], nX, cx);
  }

  kwGroupInv<KW_NDX>(dx);

  // centre itself
  Load256(xs[KW_HALF], cx);

  for (int t = 1; t <= KW_HALF; t++) {

    // negative side: centre - tG  (offset -t)
    // dy = (-tG).y - centre.y = -tG.y - centre.y
    {
      uint64_t negY[4], dy[4], s[4], s2[4], rx[4];
      Load256(negY, gyv[t - 1]);
      ModNeg256(negY);
      ModSub256(dy, negY, cy);
      _ModMult(s, dy, dx[t - 1]);
      _ModSqr(s2, s);
      ModSub256(rx, s2, cx);
      ModSub256(rx, rx, gxv[t - 1]);   // x(-tG) == x(tG)
      Load256(xs[KW_HALF - t], rx);
    }

    // positive side: centre + tG (offset +t), only needed when t < KW_HALF
    if (t < KW_HALF) {
      kwAddX(xs[KW_HALF + t], cx, cy, gxv[t - 1], gyv[t - 1], dx[t - 1]);
    }

    // the (t == KW_HALF) positive side is not part of the group: it is the
    // next centre minus 32G, produced by the next group.

  }

  // next centre = centre + KW_GROUP*G (complete addition, inv already computed)
  kwAddPoint(ncx, ncy, cx, cy, nX, nY, dx[KW_HALF]);

}

// Forward walk (+G), the one used by the xpoint scan.
KW_EMUL_STATIC __device__ void kwWalkGroup(uint64_t xs[KW_GROUP][4],
                            uint64_t *ncx, uint64_t *ncy,
                            uint64_t *cx, uint64_t *cy,
                            const uint64_t *gx, const uint64_t *gy,
                            const uint64_t *nx, const uint64_t *ny) {

  kwWalkGroupSigned(xs, ncx, ncy, cx, cy, gx, gy, nx, ny, 1);

}

#endif // GPUPOINTWALK_H
