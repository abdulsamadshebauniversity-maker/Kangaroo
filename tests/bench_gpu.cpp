/*
* Benchmark of the two GPU search modes (GPU/GPUXPoint.h, GPU/GPUBsgs.h).
*
* In -DKW_EMULATE the kernels run one thread after the other on the CPU, so the
* numbers are *not* GPU throughput: they measure how many keys the same
* algorithm processes per second, which is what the optimisation work targets
* (inversions amortised over a group, no PCIe traffic per key, one fingerprint
* per key). On a real card the same code runs the groups in parallel and the
* throughput scales with the number of resident warps.
*
*   make bench
*   obj/test/bench_gpu            (default sizes)
*   obj/test/bench_gpu 4096 64    (xpoint groups | BSGS babies)
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "SECPK1/SECP256k1.h"
#include "GPU/GPUXPoint.h"
#include "GPU/GPUBsgs.h"

int main(int argc, char **argv) {

  setvbuf(stdout, NULL, _IONBF, 0);

  uint64_t xpGroups = (argc > 1) ? strtoull(argv[1], 0, 10) : 1024;
  uint64_t nbBaby = (argc > 2) ? strtoull(argv[2], 0, 10) : 256;
  uint64_t nbGiant = (argc > 3) ? strtoull(argv[3], 0, 10) : 64;

  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  // ---------------- xpoint ----------------
  {
    Int start;
    start.SetInt32(0x400000);
    uint64_t nbKey = xpGroups * KW_GROUP;

    Int key;
    key.SetInt32(0);
    key.Set(&start);
    Int off;
    off.SetInt32((int64_t)(nbKey / 2 + 3));
    key.Add(&off);
    Point Q = secp->ComputePublicKey(&key);

    GPUXPoint xp(secp);
    xp.SetParams(start, xpGroups);
    xp.AddTarget(&Q.x);

    double t0 = kwSeconds();
    uint32_t n = xp.Search();
    double secs = kwSeconds() - t0;

    printf("xpoint : %llu keys in %.3f s -> %.0f keys/s  (found %u)\n",
           (unsigned long long)nbKey, secs, (double)nbKey / secs, n);
  }

  // ---------------- BSGS ----------------
  {
    Int start;
    start.SetInt32(0x800000);
    uint64_t interval = nbBaby * nbGiant;

    Int key;
    key.Set(&start);
    Int off;
    off.SetInt32((int64_t)(interval / 2 + 1));
    key.Add(&off);
    Point Q = secp->ComputePublicKey(&key);

    GPUBSGS bsgs(secp);
    bsgs.SetParams(start, nbBaby, nbGiant);
    bsgs.SetTarget(&Q);

    // The BSGS cost is dominated by the giant steps: the baby table is built
    // once and can be reused for every target, the giants are the search.
    // Time the whole Search() so the number is honest (baby + giants).
    double t0 = kwSeconds();
    uint32_t found = bsgs.Search() ? 1 : 0;
    double secs = kwSeconds() - t0;

    double tb, tg;
    uint64_t cand;
    bsgs.GetStats(tb, tg, cand);

    printf("bsgs   : m=%llu n=%llu interval=%llu keys in %.3f s -> %.0f keys/s "
           "(baby %.3f s, giants %.3f s; found %u, %llu candidates)\n",
           (unsigned long long)nbBaby, (unsigned long long)nbGiant,
           (unsigned long long)interval, secs, (double)interval / secs,
           tb, tg, found, (unsigned long long)cand);
  }

  delete secp;
  return 0;

}
