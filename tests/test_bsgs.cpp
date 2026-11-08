/*
* End to end test of the GPU BSGS engine (GPU/GPUBsgs.h + GPUBsgsKernel.h).
*
* The whole pipeline is exercised the way the application uses it:
*
*   GPUBSGS::SetParams(start, m, n)  -> builds the baby table on the device
*   GPUBSGS::SetTarget(Q)            -> Q = K*G
*   GPUBSGS::Search()                -> giant kernel + host verification
*
* Every run plants the key at a chosen offset of the interval and checks that
* the engine returns exactly that key (and no other): first key, last key,
* keys at a group boundary, keys in the middle of a thread slice, plus a key
* outside the interval which must not be reported. Several interval sizes and
* m/n splits are covered, including a 2^16 baby table.
*/

#include <cstdio>
#include <cstring>
#include <vector>

#include "SECPK1/SECP256k1.h"
#include "GPU/GPUBsgs.h"

static int nbTest = 0;
static int nbFail = 0;

static void check(bool ok, const char *what) {
  nbTest++;
  if (!ok) {
    nbFail++;
    printf("  [FAIL] %s\n", what);
  }
}

static Int keyAt(Secp256K1 *secp, const Int &start, uint64_t offset) {
  Int k;
  k.Set((Int *)&start);
  Int off;
  off.SetInt32((int64_t)offset);
  k.Add(&off);
  return k;
}

int main(int argc, char **argv) {

  setvbuf(stdout, NULL, _IONBF, 0);
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  printf("GPU BSGS engine tests\n");

  struct Case { uint64_t m; uint64_t n; uint64_t offset; };
  // m must stay a multiple of KW_GROUP
  struct Case cases[] = {
    {  256,     8,       0 },     // first key of the interval
    {  256,     8,     255 },     // last key of a giant
    {  256,     8,    1301 },     // middle, inside a thread slice
    {  256,     8,    2047 },     // very last key of the interval
    {  512,    32,     777 },     // bigger interval, finer split
    { 1024,     8,    8191 },     // one thread, whole interval
    { 1024,    64,    4096 },     // exact giant boundary
    { 2048,    32,   65536 - 1 }, // 2^16 keys, last one
    {  128,   128,      63 },     // more giants than babies
  };
  int nbCase = sizeof(cases) / sizeof(cases[0]);

  for (int c = 0; c < nbCase; c++) {

    Int start;
    start.SetInt32(1000000 + 7919 * c);

    uint64_t interval = cases[c].m * cases[c].n;
    Int key = keyAt(secp, start, cases[c].offset);
    Point Q = secp->ComputePublicKey(&key);

    GPUBSGS bsgs(secp);
    bsgs.SetParams(start, cases[c].m, cases[c].n);
    bsgs.SetTarget(&Q);
    bool ok = bsgs.Search();

    char label[256];
    sprintf(label, "case m=%llu n=%llu offset=%llu: key found",
            (unsigned long long)cases[c].m, (unsigned long long)cases[c].n,
            (unsigned long long)cases[c].offset);
    check(ok, label);
    if (ok) {
      Int got = bsgs.GetKey();
      sprintf(label, "case m=%llu n=%llu offset=%llu: key is exact",
              (unsigned long long)cases[c].m, (unsigned long long)cases[c].n,
              (unsigned long long)cases[c].offset);
      check(got.IsEqual(&key), label);
    }

    // A target outside the interval must not be reported
    Int outside = keyAt(secp, start, interval + 12345);
    Point Qo = secp->ComputePublicKey(&outside);
    GPUBSGS bsgs2(secp);
    bsgs2.SetParams(start, cases[c].m, cases[c].n);
    bsgs2.SetTarget(&Qo);
    sprintf(label, "case m=%llu n=%llu: key outside the interval not reported",
            (unsigned long long)cases[c].m, (unsigned long long)cases[c].n);
    check(!bsgs2.Search(), label);

  }

  printf("\n%d checks, %d failures\n", nbTest, nbFail);
  delete secp;
  return nbFail == 0 ? 0 : 1;

}
