/*
* Validation of the GPU device arithmetic (GPU/GPUMath.h) in emulation mode
* against the independent fixed size CPU implementation of SECPK1/.
*
* Every check compares the device kernel result with a result computed by
* SECPK1/Int, SECPK1/IntMod and SECPK1/SECP256K1, which are completely separate
* code paths.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// SECPK1/ first: GPUMath.h defines short macros (Neg, Load, Sub2, ...) that
// would otherwise collide with members of the CPU classes.
#include "../SECPK1/SECP256k1.h"
#include "../SECPK1/Random.h"
#include "../SECPK1/IntGroup.h"
#include "../GPU/GPUMath.h"

static int nbTest = 0;
static int nbFail = 0;

static void check(bool ok, const char *what, const std::string &detail = "") {
  nbTest++;
  if (!ok) {
    nbFail++;
    printf("  [FAIL] %s %s\n", what, detail.c_str());
  }
}

static void dump(const char *name, const uint64_t *v, int n = 5) {
  printf("    %s = ", name);
  for (int i = n - 1; i >= 0; i--) printf("%016llx", (unsigned long long)v[i]);
  printf("\n");
}

// ---------------------------------------------------------------------------

__global__ void kwTestModMult(uint64_t *a, uint64_t *b, uint64_t *r) {
  uint64_t ra[4], rb[4], rr[4];
  Load256(ra, a + threadIdx.x * 4);
  Load256(rb, b + threadIdx.x * 4);
  _ModMult(rr, ra, rb);
  Load256(r + threadIdx.x * 4, rr);
}

__global__ void kwTestModSqr(uint64_t *a, uint64_t *r) {
  uint64_t ra[4], rr[4];
  Load256(ra, a + threadIdx.x * 4);
  _ModSqr(rr, ra);
  Load256(r + threadIdx.x * 4, rr);
}

__global__ void kwTestModInv(uint64_t *a, uint64_t *r) {
  uint64_t ra[5];
  ra[0] = a[threadIdx.x * 4 + 0];
  ra[1] = a[threadIdx.x * 4 + 1];
  ra[2] = a[threadIdx.x * 4 + 2];
  ra[3] = a[threadIdx.x * 4 + 3];
  ra[4] = 0;
  _ModInv(ra);
  Load256(r + threadIdx.x * 4, ra);
}

__global__ void kwTestModInvGrouped(uint64_t *a, uint64_t *r) {
  // One group per launch, GPU_GRP_SIZE values
  uint64_t v[GPU_GRP_SIZE][4];
  for (int i = 0; i < GPU_GRP_SIZE; i++)
    Load256(v[i], a + i * 4);
  _ModInvGrouped(v);
  for (int i = 0; i < GPU_GRP_SIZE; i++)
    Load256(r + i * 4, v[i]);
}

__global__ void kwTestModSub(uint64_t *a, uint64_t *b, uint64_t *r) {
  uint64_t ra[4], rb[4], rr[4];
  Load256(ra, a + threadIdx.x * 4);
  Load256(rb, b + threadIdx.x * 4);
  ModSub256(rr, ra, rb);
  Load256(r + threadIdx.x * 4, rr);
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {

  setvbuf(stdout,NULL,_IONBF,0);
  int nb = (argc > 1) ? atoi(argv[1]) : 60;

  Secp256K1 *secp = new Secp256K1();
  secp->Init();
  rseed(0x1234567ULL);

  printf("GPUMath emulation tests (%d random vectors per test)\n", nb);

  // P (field characteristic) as Int
  Int P;
  P.SetBase16((char *)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  Int order;
  order.SetBase16((char *)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  std::vector<Int> va, vb;
  for (int i = 0; i < nb; i++) {
    Int a, b;
    a.Rand(&P);
    b.Rand(&P);
    va.push_back(a);
    vb.push_back(b);
  }

  // --- _ModMult -----------------------------------------------------------
  {
    std::vector<uint64_t> ha(nb * 4), hb(nb * 4), hr(nb * 4);
    for (int i = 0; i < nb; i++) {
      memcpy(&ha[i * 4], va[i].bits64, 32);
      memcpy(&hb[i * 4], vb[i].bits64, 32);
    }
    KW_LAUNCH(kwTestModMult, 1, nb, ha.data(), hb.data(), hr.data());
    for (int i = 0; i < nb; i++) {
      Int r;
      r.ModMulK1(&va[i], &vb[i]);
      bool ok = memcmp(r.bits64, &hr[i * 4], 32) == 0;
      check(ok, "ModMult");
      if (!ok && nbFail < 5) { dump("dev", &hr[i * 4]); dump("cpu", r.bits64); }
    }
    printf("  ModMult           : %d vectors\n", nb);
  }

  // --- _ModSqr ------------------------------------------------------------
  {
    std::vector<uint64_t> ha(nb * 4), hr(nb * 4);
    for (int i = 0; i < nb; i++)
      memcpy(&ha[i * 4], va[i].bits64, 32);
    KW_LAUNCH(kwTestModSqr, 1, nb, ha.data(), hr.data());
    for (int i = 0; i < nb; i++) {
      Int r;
      r.ModSquareK1(&va[i]);
      bool ok = memcmp(r.bits64, &hr[i * 4], 32) == 0;
      check(ok, "ModSqr");
      if (!ok && nbFail < 5) { dump("dev", &hr[i * 4]); dump("cpu", r.bits64); }
    }
    printf("  ModSqr            : %d vectors\n", nb);
  }

  // --- _ModInv ------------------------------------------------------------
  {
    std::vector<uint64_t> ha(nb * 4), hr(nb * 4);
    for (int i = 0; i < nb; i++)
      memcpy(&ha[i * 4], va[i].bits64, 32);
    KW_LAUNCH(kwTestModInv, 1, nb, ha.data(), hr.data());
    for (int i = 0; i < nb; i++) {
      Int r;
      r.Set(&va[i]);
      r.ModInv();
      bool ok = memcmp(r.bits64, &hr[i * 4], 32) == 0;
      check(ok, "ModInv");
      if (!ok && nbFail < 5) { dump("dev", &hr[i * 4]); dump("cpu", r.bits64); }
    }
    printf("  ModInv            : %d vectors\n", nb);
  }

  // --- _ModInvGrouped -----------------------------------------------------
  {
    // Grouped inversion: one modular inverse per group of GPU_GRP_SIZE
    // values, computed with a single inversion + 3n multiplications.
    std::vector<uint64_t> ha(GPU_GRP_SIZE * 4), hr(GPU_GRP_SIZE * 4);
    std::vector<Int> va;
    for (int i = 0; i < GPU_GRP_SIZE; i++) {
      Int v;
      v.SetInt32(1);
      v.Rand(&P);
      va.push_back(v);
      memcpy(&ha[i * 4], v.bits64, 32);
    }
    KW_LAUNCH(kwTestModInvGrouped, 1, 1, ha.data(), hr.data());
    IntGroup grp(GPU_GRP_SIZE);
    Int *oracle = new Int[GPU_GRP_SIZE];
    for (int i = 0; i < GPU_GRP_SIZE; i++) oracle[i].Set(&va[i]);
    grp.Set(oracle);
    grp.ModInv();
    for (int i = 0; i < GPU_GRP_SIZE; i++) {
      bool ok = memcmp(oracle[i].bits64, &hr[i * 4], 32) == 0;
      check(ok, "ModInvGrouped");
      if (!ok && nbFail < 8) { dump("dev", &hr[i * 4]); dump("cpu", oracle[i].bits64); }
      // a * a^-1 must be congruent to 1 mod P. SECPK1/Int::ModMulK1 uses a
      // lazy reduction (result in [0,2P)), so accept 1 and P+1.
      Int inv, chk, one, pPlus1, P2;
      inv.SetInt32(0);
      memcpy(inv.bits64, &hr[i * 4], 32);
      chk.ModMulK1(&va[i], &inv);
      one.SetInt32(1);
      P2.Set(&P);
      pPlus1.Set(&P);
      pPlus1.Add(&one);
      check(chk.IsEqual(&one) || chk.IsEqual(&pPlus1), "ModInvGrouped a*inv==1");
    }
    delete[] oracle;
    printf("  ModInvGrouped     : %d values\n", GPU_GRP_SIZE);
  }

  // --- Add / Sub / Neg / Mult with the order and with P --------------------
  {
    const int N = 16;
    std::vector<uint64_t> ha(N * 4), hb(N * 4), hr(N * 4);
    std::vector<Int> sa, sb;
    for (int i = 0; i < N; i++) {
      Int a, b;
      a.Rand(&P);
      b.Rand(&P);
      sa.push_back(a);
      sb.push_back(b);
      memcpy(&ha[i * 4], a.bits64, 32);
      memcpy(&hb[i * 4], b.bits64, 32);
    }
    KW_LAUNCH(kwTestModSub, 1, N, ha.data(), hb.data(), hr.data());
    for (int i = 0; i < N; i++) {
      Int r;
      r.ModSub(&sa[i], &sb[i]);
      check(memcmp(r.bits64, &hr[i * 4], 32) == 0, "ModSub256");
    }
    printf("  ModSub256         : %d vectors\n", N);
  }

  printf("\n%d checks, %d failures\n", nbTest, nbFail);
  delete secp;
  return nbFail == 0 ? 0 : 1;

}
