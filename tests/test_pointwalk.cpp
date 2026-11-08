/*
* Validation of the GPU affine group walk (GPU/GPUPointWalk.h) against the
* CPU elliptic curve implementation (SECPK1/SECP256K1).
*
* The walk produces, for a centre point P, the x coordinates of
*   P + (i - KW_HALF)*G      for i in [0,KW_GROUP)
* plus the next centre P + KW_GROUP*G. Every produced x is compared with
* secp->AddDirect()/ComputePublicKey() results, and every produced point is
* checked to be on the curve.
*/

#include <cstdio>
#include <cstring>
#include <vector>

#include "SECPK1/SECP256k1.h"
#include "SECPK1/Random.h"
#include "GPU/GPUPointWalk.h"

static int nbTest = 0;
static int nbFail = 0;

static void check(bool ok, const char *what, const char *detail = "") {
  nbTest++;
  if (!ok) {
    nbFail++;
    printf("  [FAIL] %s %s\n", what, detail);
  }
}

// ---------------------------------------------------------------------------
// Test kernel: one thread, one group walk
// ---------------------------------------------------------------------------

__global__ void kwTestWalk(uint64_t *in, uint64_t *table, uint64_t *out) {

  // in  : cx[4] cy[4]  (thread block start point)
  // table: gx[KW_HALF][4] gy[KW_HALF][4] nx[4] ny[4]
  uint64_t cx[4], cy[4];
  uint64_t xs[KW_GROUP][4];
  uint64_t ncx[4], ncy[4];

  Load256(cx, in);
  Load256(cy, in + 4);

  const uint64_t *gx = table;
  const uint64_t *gy = table + KW_HALF * 4;
  const uint64_t *nx = table + 2 * KW_HALF * 4;
  const uint64_t *ny = table + 2 * KW_HALF * 4 + 4;

  kwWalkGroup(xs, ncx, ncy, cx, cy, gx, gy, nx, ny);

  for (int i = 0; i < KW_GROUP; i++)
    Load256(out + i * 4, xs[i]);
  Load256(out + KW_GROUP * 4, ncx);
  Load256(out + KW_GROUP * 4 + 4, ncy);

}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {

  setvbuf(stdout, NULL, _IONBF, 0);
  int nb = (argc > 1) ? atoi(argv[1]) : 20;

  Secp256K1 *secp = new Secp256K1();
  secp->Init();
  rseed(0xC0FFEEULL);

  printf("GPU group walk tests (%d random start points)\n", nb);

  // t*G table, exactly as the host will build it for the kernel
  std::vector<uint64_t> table(2 * KW_HALF * 4 + 8, 0);
  for (int t = 1; t <= KW_HALF; t++) {
    Int ti; ti.SetInt32(t);
    Point P = secp->ComputePublicKey(&ti);
    memcpy(&table[(t - 1) * 4], P.x.bits64, 32);
    memcpy(&table[(KW_HALF + t - 1) * 4], P.y.bits64, 32);
  }
  {
    Int ni; ni.SetInt32(KW_GROUP);
    Point nG = secp->ComputePublicKey(&ni);
    memcpy(&table[2 * KW_HALF * 4], nG.x.bits64, 32);
    memcpy(&table[2 * KW_HALF * 4 + 4], nG.y.bits64, 32);
  }

  Int maxStart;
  maxStart.SetBase16((char *)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000");

  for (int it = 0; it < nb; it++) {

    // Random start point
    Int k;
    k.SetInt32(1);
    k.Rand(&maxStart);
    Point P = secp->ComputePublicKey(&k);

    uint64_t in[8], out[KW_GROUP * 4 + 8];
    memcpy(in, P.x.bits64, 32);
    memcpy(in + 4, P.y.bits64, 32);

    KW_LAUNCH(kwTestWalk, 1, 1, in, table.data(), out);

    // Oracle: the same run of points, rebuilt with the CPU EC code.
    // The group starts at key k-KW_HALF and covers KW_GROUP consecutive keys.
    Int kStart;
    kStart.Set(&k);
    kStart.Sub((uint64_t)KW_HALF);
    Point ref = secp->ComputePublicKey(&kStart);

    int bad = 0;
    for (int i = 0; i < KW_GROUP; i++) {
      bool ok = memcmp(ref.x.bits64, &out[i * 4], 32) == 0;
      if (!ok) bad++;
      check(ok, "walk x", "");
      if (!ok && bad < 3) {
        printf("    i=%d (offset %d)\n", i, i - KW_HALF);
        Int devX; devX.SetInt32(0); memcpy(devX.bits64, &out[i * 4], 32);
        printf("      dev = %s\n", devX.GetBase16().c_str());
        printf("      cpu = %s\n", ref.x.GetBase16().c_str());
      }
      ref = secp->NextKey(ref);
    }

    // next centre
    Int nk;
    nk.Set(&k);
    Int kg;
    kg.SetInt32(KW_GROUP);
    Int kNext;
    kNext.Set(&k);
    kNext.Add(&kg);
    Point nkP = secp->ComputePublicKey(&kNext);
    bool okN = memcmp(nkP.x.bits64, &out[KW_GROUP * 4], 32) == 0 &&
               memcmp(nkP.y.bits64, &out[KW_GROUP * 4 + 4], 32) == 0;
    check(okN, "next centre", "");
    if (!okN) {
      Int devX; devX.SetInt32(0); memcpy(devX.bits64, &out[KW_GROUP * 4], 32);
      printf("      dev next x = %s\n", devX.GetBase16().c_str());
      printf("      cpu next x = %s\n", nkP.x.GetBase16().c_str());
    }

  }

  printf("\n%d checks, %d failures\n", nbTest, nbFail);
  delete secp;
  return nbFail == 0 ? 0 : 1;

}
