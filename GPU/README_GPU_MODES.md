# GPU (CUDA) port of the Keyhunt Xpoint and BSGS modes

This document is the engineering record of the port of the two *Keyhunt*
search modes (**Xpoint** and **BSGS**) from the CPU implementation to CUDA
kernels running on top of the existing device arithmetic of this repository
(`GPU/GPUMath.h`, the same 256 bit SECP256K1 math the Kangaroo kernels use).

Everything described here is in the tree and exercised by `make check`.

```
GPU/GPUCompat.h        device/host portability + SIMT emulator + CUDA runtime shim
GPU/GPUMath.h          existing device arithmetic (mod mult/sqr/inv/group inv, PTX)
GPU/GPUPointWalk.h     affine group walk shared by both modes, 64 bit fingerprint
GPU/GPUXPointKernel.h  device side of the xpoint scan
GPU/GPUXPoint.{h,cpp}  host driver of the xpoint scan
GPU/GPUBsgsKernel.h    device side of BSGS (babies, fingerprints, giants)
GPU/GPUBsgs.{h,cpp}    host driver of BSGS
GPU/GPULaunch.h        the only place that launches a kernel (nvcc <<<>>> or emulator)
GPUModes.cpp           command line front end (--mode xpoint | --mode bsgs)
tests/test_xpoint.cpp  xpoint kernel + driver tests
tests/test_bsgs.cpp    BSGS pipeline tests
tests/bench_gpu.cpp    throughput of both modes
```

---

## 1. Analysis of the CPU reference

### 1.1 Xpoint mode (`MODE_XPOINT` in keyhunt)

For every key of the interval the CPU version computes the x coordinate, feeds
it to a *bloom filter* of the target x values and, for the rare positive, does
a full check. The interesting part is how the x coordinates are produced:
keyhunt walks the interval with `IntGroup` — a batch of 64 consecutive keys is
produced from **one** modular inversion plus 64 x-only additions, instead of 64
scalar multiplications.

### 1.2 BSGS mode (`MODE_BSGS`)

Interval `[start, start + m*n)`, target `Q = K*G`:

```
babies : b_i = (start + i)*G          i = 0 .. m-1     -> sorted/hashed table
giants : B_j = Q - (j*m)*G            j = 0 .. n-1
match  : B_j == b_i    <=>   K = start + j*m + i
```

The CPU version builds `n` bloons filters (one per giant, each holding the
results of `Q - (j*m)*G` shifted by `-i*G`, i.e. the whole `m` neighbourhood of
that giant) and then scans the babies once. It uses `IntGroup` for both halves.

### 1.3 What has to change for the GPU

| CPU | GPU problem | decision |
|---|---|---|
| one thread does everything | 2^23 keys need ~10^5 parallel threads | one CUDA thread owns a contiguous slice of the interval |
| bloom filter built on the host | per-key traffic to compare against it | 64 bit fingerprint of x computed on the device, kept in registers |
| x coordinates never leave the CPU | 32 bytes/key crossing PCIe | only matches are copied back (44 bytes each) |
| `IntGroup` (inversion per 64 keys) | recompute the inversion on the GPU | `kwGroupInv` — batch inversion, 1 inversion + 3 multiplies per value |
| scalar multiplication per key | way too slow | affine group walk: 64 additions per group, 1 inversion amortised |

---

## 2. Design

### 2.1 The affine group walk (`GPU/GPUPointWalk.h`)

A *group* is 64 consecutive keys around a centre point `P`:

```
xs[i] = x(P + (i-32)*G)      i = 0..63        xs[32] = x(P)
next centre = P + 64*G
```

Cost per 64 keys:

* 33 differences `x(tG) - x(P)` (t = 1..32 and t = 64) with `t*G` from a
  constant table, then **one** inversion for all of them (`kwGroupInv`),
* 64 x-only additions (`kwAddX`, one mod mult + one mod sqr each),
* one complete addition for the next centre (`kwAddPoint`, uses the inversion
  of `x(64G) - x(P)` already computed).

Both modes use exactly this primitive, which is why the same table layout
(`gTable`: `t*G` for t = 1..32, then `64*G`) is used by both kernels.

### 2.2 Matching

x coordinates are compared through a 64 bit fingerprint (`kwFingerprint`,
a splitmix-style mixer, bit identical on host and device). A 64 bit match is
only a *candidate*; the host driver re-derives the key with a full scalar
multiplication before reporting it, so

* a collision can never produce a wrong key,
* the probability of a *missed* key is zero (the fingerprint is a function of
  the exact x, every key's x is hashed).

### 2.3 Xpoint kernel

```
thread tid -> groups [tid*groupPerThread, +groupPerThread)
   centre = host-precomputed P for the first group of the slice
   loop over its groups:
     kwWalkGroup -> 64 x -> 64 fingerprints -> compare with the <=32 targets in registers
     centre += 64G  (the walk already produced it)
```

Result buffer: `[0]` = atomic match count, then `KW_MATCH_U32 = 11` words per
match (absolute key, x, fingerprint). Only the counter and the matches are read
back.

### 2.4 BSGS kernels

```
1. kwBabyKernel       : walk the m babies (same group walk), store their x
2. kwFingerprintKernel: fold those x into fingerprints *on the device*
3. host counting sort : CSR bucket table (bucketStart prefix sums)
4. kwGiantKernel      : B_j in registers, step by -m*G, one fingerprint +
                        bucket lookup per giant, atomic match list
5. host verification  : recompute the key, accept only Q == key*G
```

Two design decisions are worth recording because the first attempts got them
wrong:

* **the giant steps are not a walk.** Shifting a giant by `t*G` just gives
  another giant index, so a `±32` sweep around every giant is structurally
  wrong (it reports matches at the wrong index). Each giant is compared once.
* **the baby table must be exact.** A fixed 4-entries-per-bucket table silently
  drops candidates when a bucket overflows (the tests found it as a "missing
  key" with one key per 20 runs). The CSR/counting-sort table has no capacity
  and therefore no failure mode.

Cost model: the babies are built once and can be reused for any number of
targets; the giants dominate and cost 1 inversion + 1 fingerprint per `m`
keys (the step `-m*G`), which is why BSGS outruns the xpoint scan by roughly
an order of magnitude per key on the same hardware.

### 2.5 Host/device boundary

`GPU/GPULaunch.h` is the only place that knows about `<<< >>>`; the four
`kwLaunchXxx()` inline functions forward to `KW_LAUNCH`:

* `-DKW_EMULATE` → `KW_LAUNCH` runs the kernel through the SIMT emulator of
  `GPUCompat.h` (no CUDA toolchain, no GPU);
* real CUDA → `KW_LAUNCH` *is* the kernel launch.

Because the drivers include the kernel headers, the two of them
(`GPU/GPUXPoint.cpp`, `GPU/GPUBsgs.cpp`) are compiled by `nvcc` in the GPU
build and by g++ in the emulated build - the same source, two compilers, no
duplicated marshalling code.

`GPUCompat.h` also provides a small CUDA runtime shim under emulation
(`cudaMalloc`/`cudaMemcpy`/`cudaMemset`/`cudaDeviceSynchronize`), so the driver
source is identical in both worlds.

---

## 3. Implementation notes (bugs found by the tests)

The unit tests are not decoration: they found five real defects.

1. **PTX carry semantics** — `mad.hi.cc` does not consume the carry-in;
   `madc.hi.cc` / `madc.hi.u64` do (`madc.hi.s64` is signed). The emulator now
   mirrors this exactly, which is why the 464 arithmetic checks are meaningful.
2. **`_ModInv` works on 5 limbs** — calling it on a `uint64_t[4]` reads a
   garbage carry limb. All inversions go through `kwInv` (zeroed 5th limb).
3. **Aliasing in the group walk** — `kwWalkGroup(xs, cx, cy, cx, cy, ...)`
   aliases the new centre with the input. `kwAddPoint` now accumulates in
   temporaries and stores at the end; before that fix the first group of every
   slice was correct and the chain after it was garbage.
4. **Strided thread mapping** — the walk advances the centre by exactly one
   group, so the interval must be split into *contiguous* slices (a
   `g += nbThreads` mapping computes wrong keys).
5. **Giant walk vs giant index** and **bucket overflow** (see 2.4).

`kwWalkGroup` also grew a `sign` parameter: the giants step backwards
(`-m*G`), and rather than duplicating the walk the negative direction negates
the precomputed `t*G` y coordinates.

---

## 4. Tests

```
make check
```

| target | what it proves | checks |
|---|---|---|
| `check-arithmetic` | all 256 bit primitives against `SECPK1/` (mult, sqr, inv, grouped inv, sub, mod neg, IMult, ctz, byte perm…) | 464 |
| `check-pointwalk` | the group walk: 64 x values + next centre x/y against `ComputePublicKey`/`NextKey`, 5 random starts | 325 |
| `check-xpoint` | the scan kernel and `GPUXPoint` for six launch geometries: first/last key of the interval, both sides of a slice boundary, a key outside (must not match), driver API | 41 |
| `check-bsgs` | the whole BSGS pipeline (baby walk, fingerprints, CSR table, giants, host verification) for nine `(m, n, offset)` cases up to a 2^16 key interval, including the first/last key and one key outside | 27 |

`make bench` prints the throughput of both modes (see 5).

All of this runs through `-DKW_EMULATE`, i.e. the **real device sources** are
compiled by g++, with the PTX carry chains emulated bit exactly and the kernels
executed thread by thread.

---

## 5. Benchmark (emulated run, one CPU core)

```
$ make bench
xpoint : 65536 keys in 0.013 s -> 5107261 keys/s  (found 1)
bsgs   : m=256 n=64 interval=16384 keys in 0.001 s -> 25135141 keys/s (found 1, 1 candidate)

$ obj/test/bench_gpu 8192 512 256
xpoint : 524288 keys in 0.089 s -> 5861905 keys/s  (found 1)
bsgs   : m=512 n=256 interval=131072 keys in 0.002 s -> 57010869 keys/s (found 1, 1 candidate)
```

These numbers are **per core, single threaded**: the emulator executes the
grid sequentially. They measure the algorithmic work per key (the part the port
can control):

* ~1 inversion per 64 keys instead of 1 per key,
* no per-key memory traffic and no per-key PCIe traffic,
* BSGS: 1 inversion per `m` keys.

On a GPU the same code runs the threads in parallel; the throughput then scales
with the number of resident warps, and the numbers should be compared against
the CPU `keyhunt` modes rather than against these single-core figures. A
GPU benchmark must be run on a machine with a CUDA device (see 6).

---

## 6. Building with a real GPU

```bash
make gpu=1 ccap=86          # ccap = compute capability of your card
```

* `CUDA`, `NVCC`, `CXXCUDA` (host compiler for nvcc) and `ccap` are variables;
  the old hardcoded `cuda-8.0` + `g++-4.8` pair was removed, it does not exist
  on current systems and cannot build the C++11 headers of the project.
* `GPU/GPUXPoint.cpp` and `GPU/GPUBsgs.cpp` are compiled by `nvcc` (they include
  the kernel headers), the rest of the program stays plain g++.
* `make check` and `make emu` build the emulator variant, which does not need
  CUDA at all - `make emu` produces `kangaroo-emu`, a fully usable binary.

### What the emulation does not cover

The emulator runs the grid one thread at a time, so it verifies **arithmetic,
memory layout and algorithm** but not **concurrency**: warp divergence,
`atomicAdd` contention on the match list, or a bucket table being read while
written. The kernels are written so that every thread owns its slice, touches
only its own state and communicates exclusively through `atomicAdd` on the match
counter, but a run on real hardware remains mandatory before production use.

---

## 7. Using the modes

From the command line (`make emu` for a CPU-only binary, `make gpu=1` for CUDA):

```bash
# xpoint scan of 200000 keys starting at 0x120000, one x target
./kangaroo --mode xpoint --start 0x120000 --range 200000 \
           --x 8208F5AB...782F

# BSGS over the same interval, 4096 babies
./kangaroo --mode bsgs --start 0x120000 --range 200000 \
           --pubkey 04 8208F5AB...782F B7A0AC7E...D4DA --m 4096
```

Both print `key=<hex> x=<hex>` per verified match; `make check-cli` runs
exactly that as a smoke test.

### 7.1 Using the engines

### Xpoint (GPU/GPUXPoint.h)

```cpp
GPUXPoint xp(secp);
xp.SetParams(startKey, nbGroup);      // interval = nbGroup * 64 keys
xp.AddTarget(&target.x);              // up to KW_MAX_TARGETS (32) x coordinates
uint32_t n = xp.Search();
for (uint32_t i = 0; i < n; i++)
  printf("key %s\n", xp.GetFoundKey(i).GetHex().c_str());
```

### BSGS (GPU/GPUBsgs.h)

```cpp
GPUBSGS bsgs(secp);
bsgs.SetParams(startKey, nbBaby, nbGiant);   // interval = nbBaby * nbGiant
bsgs.SetTarget(&Q);                          // Q = K*G
if (bsgs.Search()) {
  Int K = bsgs.GetKey();                     // absolute key
  // bsgs.GetStats(secsBaby, secsGiant, nbCandidates);
}
```

Both drivers own their device buffers and re-verify every candidate on the
host before reporting it.

---

## 8. Status / roadmap

* [x] device arithmetic verified against `SECPK1/`
* [x] affine group walk verified
* [x] xpoint kernel + driver verified (all launch geometries)
* [x] BSGS kernels + driver verified (baby table, exact bucket table, giants,
      host verification)
* [x] build integration (`GPU/GPULaunch.cu`, Makefile autodetection)
* [x] tests and benchmark in `make check` / `make bench`
* [ ] run `make gpu=1 ccap=XX` and the benchmark on a real CUDA device
* [ ] add a command line front end (`--mode xpoint` / `--mode bsgs`) to
      `main.cpp` so the modes are usable from the binary
* [ ] feed the matches into the existing `Check.cpp` / `Network.cpp` pipeline
