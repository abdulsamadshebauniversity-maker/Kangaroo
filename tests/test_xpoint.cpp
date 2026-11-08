/*
* End to end test of the GPU xpoint search kernel (GPU/GPUXPointKernel.h).
*
* The kernel scans a key interval with the affine group walk and reports the
* keys whose x coordinate fingerprint matches one of the targets. The test
* builds the interval, plants targets at known offsets (including the first
* and the last key of the interval, and keys belonging to different thread
* slices) plus one key outside the interval, runs the kernel with several
* launch geometries and checks that exactly the planted in-range keys are
* found, with the correct x coordinate for each of them.
*/

#include <cstdio>
#include <cstring>
#include <vector>

#include "SECPK1/SECP256k1.h"
#include "SECPK1/Random.h"
#include "GPU/GPUXPointKernel.h"

static int nbTest = 0;
static int nbFail = 0;

static void check(bool ok, const char *what) {
  nbTest++;
  if (!ok) {
    nbFail++;
    printf("  [FAIL] %s\n", what);
  }
}

// Build the t*G / KW_GROUP*G table used by the kernel
static void buildWalkTable(Secp256K1 *secp, std::vector<uint64_t> &t) {
  t.assign(2 * KW_HALF * 4 + 8, 0);
  for (int k = 1; k <= KW_HALF; k++) {
    Int i; i.SetInt32(k);
    Point P = secp->ComputePublicKey(&i);
    memcpy(&t[(k - 1) * 4], P.x.bits64, 32);
    memcpy(&t[(KW_HALF + k - 1) * 4], P.y.bits64, 32);
  }
  Int n; n.SetInt32(KW_GROUP);
  Point nG = secp->ComputePublicKey(&n);
  memcpy(&t[2 * KW_HALF * 4], nG.x.bits64, 32);
  memcpy(&t[2 * KW_HALF * 4 + 4], nG.y.bits64, 32);
}

// Run one scan and return the list of (key, x) matches
struct Match { uint64_t key; uint64_t x[4]; };

static std::vector<Match> runScan(Secp256K1 *secp, uint64_t rangeStart, uint64_t nbGroup,
                                  int grid, int block, uint64_t groupPerThread,
                                  const std::vector<uint64_t> &targetX,
                                  uint32_t maxFound) {

  std::vector<uint64_t> table;
  buildWalkTable(secp, table);

  uint64_t nbThread = (uint64_t)grid * block;

  // Centres of the first group of each thread: key = rangeStart +
  // tid*groupPerThread*KW_GROUP + KW_HALF
  std::vector<uint64_t> kang(nbThread * 8, 0);
  for (uint64_t i = 0; i < nbThread; i++) {
    Int k;
    k.SetInt32((int64_t)(rangeStart + i * groupPerThread * KW_GROUP + KW_HALF));
    Point P = secp->ComputePublicKey(&k);
    memcpy(&kang[i * 8], P.x.bits64, 32);
    memcpy(&kang[i * 8 + 4], P.y.bits64, 32);
  }

  // fingerprints of the target x values
  std::vector<uint64_t> targets;
  for (size_t i = 0; i < targetX.size() / 4; i++) {
    // targetX holds the 4 limbs of the x coordinate, packed
    targets.push_back(kwFingerprint(&targetX[i * 4]));
  }

  std::vector<uint32_t> results(1 + maxFound * KW_MATCH_U32, 0);
  uint64_t cursor = 0;

  KWScanParams p;
  p.rangeStart = rangeStart;
  p.nbGroup = nbGroup;
  p.groupPerThread = groupPerThread;
  p.nbTarget = (uint32_t)targets.size();
  p.maxFound = maxFound;

  KW_LAUNCH(kwXPointKernel, grid, block, kang.data(), table.data(), targets.data(),
            results.data(), p, &cursor);

  std::vector<Match> found;
  uint32_t n = results[0];
  if (n > maxFound) n = maxFound;
  for (uint32_t m = 0; m < n; m++) {
    uint32_t *e = &results[1 + m * KW_MATCH_U32];
    Match mt;
    mt.key = (uint64_t)e[0] | ((uint64_t)e[1] << 32);
    mt.x[0] = (uint64_t)e[2] | ((uint64_t)e[3] << 32);
    mt.x[1] = (uint64_t)e[4] | ((uint64_t)e[5] << 32);
    mt.x[2] = (uint64_t)e[6] | ((uint64_t)e[7] << 32);
    mt.x[3] = (uint64_t)e[8] | ((uint64_t)e[9] << 32);
    found.push_back(mt);
  }
  return found;

}

int main(int argc, char **argv) {

  setvbuf(stdout, NULL, _IONBF, 0);
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  printf("GPU xpoint kernel tests\n");

  const uint64_t rangeStart = 1000;
  const int geometries[][3] = { {1,1,1}, {2,2,3}, {4,4,5}, {3,1,7}, {1,8,2}, {5,3,2} };

  for (int gi = 0; gi < 6; gi++) {

    int grid = geometries[gi][0];
    int block = geometries[gi][1];
    uint64_t gpt = (uint64_t)geometries[gi][2];
    uint64_t nbGroup = (uint64_t)grid * block * gpt;
    uint64_t nbKey = nbGroup * KW_GROUP;

    // Targets: first key, last key, and keys in the middle of different slices
    std::vector<uint64_t> keys;
    keys.push_back(rangeStart);
    keys.push_back(rangeStart + nbKey - 1);
    keys.push_back(rangeStart + nbKey / 3);
    keys.push_back(rangeStart + nbKey / 2);
    if (grid * block > 1)
      keys.push_back(rangeStart + gpt * KW_GROUP + 5);     // second thread slice
    std::vector<uint64_t> tX;
    for (size_t i = 0; i < keys.size(); i++) {
      Int k; k.SetInt32((int64_t)keys[i]);
      Point P = secp->ComputePublicKey(&k);
      tX.resize(tX.size() + 4);
      memcpy(&tX[tX.size() - 4], P.x.bits64, 32);
    }
    // one key outside the interval (must not be reported)
    {
      Int k; k.SetInt32((int64_t)(rangeStart + nbKey + 12345));
      Point P = secp->ComputePublicKey(&k);
      tX.resize(tX.size() + 4);
      memcpy(&tX[tX.size() - 4], P.x.bits64, 32);
    }

    std::vector<Match> found = runScan(secp, rangeStart, nbGroup, grid, block, gpt, tX, 32);

    char label[128];
    sprintf(label, "geometry grid=%d block=%d gpt=%llu finds all in-range targets",
            grid, block, (unsigned long long)gpt);
    check(found.size() == keys.size(), label);

    for (size_t i = 0; i < keys.size(); i++) {
      bool ok = false;
      for (size_t m = 0; m < found.size(); m++) {
        if (found[m].key == keys[i]) {
          Int k; k.SetInt32((int64_t)keys[i]);
          Point P = secp->ComputePublicKey(&k);
          ok = memcmp(P.x.bits64, found[m].x, 32) == 0;
        }
      }
      sprintf(label, "geometry %d: key %llu found with the right x", gi, (unsigned long long)keys[i]);
      check(ok, label);
    }

  }

  // No target at all: nothing must be reported
  {
    std::vector<uint64_t> empty;
    std::vector<Match> found = runScan(secp, rangeStart, 4, 1, 2, 2, empty, 8);
    check(found.empty(), "no target -> no match");
  }

  printf("\n%d checks, %d failures\n", nbTest, nbFail);
  delete secp;
  return nbFail == 0 ? 0 : 1;

}
