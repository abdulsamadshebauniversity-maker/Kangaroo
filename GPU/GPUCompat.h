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
* GPUCompat.h - portability layer for the 256 bit SECP256K1 device math.
*
* The device code of this project (GPU/GPUMath.h and the scanning kernels) can be
* compiled in two ways:
*
*   1. KW_HAVE_PTX_ASM == 1  (default under nvcc)
*      Everything is a normal CUDA device function and the 64x64->128 bit
*      primitives use the inline PTX of GPUMath.h. This is the fast production
*      path.
*
*   2. KW_EMULATE == 1  (host compiler, -DKW_EMULATE)
*      The exact same sources are compiled by g++/clang and executed as a
*      single threaded SIMT emulator: __device__/__global__ vanish, threadIdx
*      and blockIdx are globals driven by kwEmulateKernel(), and the PTX
*      add/sub/mul-with-carry primitives are emulated bit exactly with
*      __int128 and a carry flag.
*
*      This gives us a way to run unit tests and full algorithm end to end
*      tests of the *device code* on a machine without GPU, and to prove the
*      GPU algorithms against the independent CPU implementation in SECPK1/.
*
* The emulation is bit exact for the integer core (carry chains included), so a
* test that passes in emulation exercises the very same arithmetic the GPU runs.
* It does not model warp scheduling, races or performance.
* ---------------------------------------------------------------------------------
*/

#ifndef GPUCOMPAT_H
#define GPUCOMPAT_H

// ---------------------------------------------------------------------------------
// Mode selection
// ---------------------------------------------------------------------------------

#if defined(__CUDACC__)
#ifndef KW_HAVE_PTX_ASM
#define KW_HAVE_PTX_ASM 1
#endif
#endif

#if !defined(KW_HAVE_PTX_ASM)
#define KW_HAVE_PTX_ASM 0
#endif

#if KW_HAVE_PTX_ASM
#if defined(KW_EMULATE)
#error "KW_EMULATE cannot be used together with the CUDA PTX backend"
#endif
#endif

#if !KW_HAVE_PTX_ASM && !defined(KW_EMULATE)
#define KW_EMULATE 1
#endif

// Marker used for functions that must be compiled by both nvcc and the host
// compiler (shared device/host helpers such as the fingerprint hash).
#if KW_EMULATE
#define KW_HOST_DEVICE inline
#else
#define KW_HOST_DEVICE __host__ __device__ inline
#endif

// ---------------------------------------------------------------------------------
// CUDA backend: nothing to emulate, only the shared markers stay.
// ---------------------------------------------------------------------------------

#if KW_HAVE_PTX_ASM

#include <cstdint>
#include <cstddef>

#ifndef KW_DEVICE_INLINE
#define KW_DEVICE_INLINE __device__ __forceinline__
#endif

#else // ---------------------------------------------------------------- emulation --

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define __device__
#define __host__
#define __global__
// In emulation each translation unit keeps its own copy of the constant
// memory arrays (internal linkage) so the header can be included by several
// test/host modules without duplicate symbol errors.
#define __constant__ static
#define __shared__ static
#define __forceinline__ inline
#define __noinline__
#define __restrict__
#define __launch_bounds__(...)
#define KW_DEVICE_INLINE inline

// Barriers are meaningless for the single threaded simulator. The scanning
// kernels are written so that every thread only touches its own state.
#define __syncthreads() ((void)0)

// ---------------------------------------------------------------------------------
// Grid / block emulation
// ---------------------------------------------------------------------------------

struct kwEmulDim3 {
  unsigned int x, y, z;
};

static kwEmulDim3 kw_emul_threadIdx = { 0, 0, 0 };
static kwEmulDim3 kw_emul_blockIdx = { 0, 0, 0 };
static kwEmulDim3 kw_emul_blockDim = { 1, 1, 1 };
static kwEmulDim3 kw_emul_gridDim = { 1, 1, 1 };

#define threadIdx kw_emul_threadIdx
#define blockIdx  kw_emul_blockIdx
#define blockDim  kw_emul_blockDim
#define gridDim   kw_emul_gridDim

// ---------------------------------------------------------------------------------
// Integer helpers provided by the CUDA runtime
// ---------------------------------------------------------------------------------

inline unsigned int kw_emul_clzll(uint64_t x) {
  return (unsigned int)(x ? __builtin_clzll(x) : 64);
}
inline unsigned int kw_emul_clz32(uint32_t x) {
  return (unsigned int)(x ? __builtin_clz(x) : 32);
}
inline uint64_t kw_emul_brevll(uint64_t x) {
  return __builtin_bswap64(x);
}

#define __clzll(x) kw_emul_clzll(x)
#define __clz(x)   kw_emul_clz32(x)
#define __brevll(x) kw_emul_brevll(x)
#define __ldg(p)   (*(p))
#define __byte_perm(a,b,s) kw_emul_byte_perm((a),(b),(s))
#define __threadfence() ((void)0)
#define __syncwarp() ((void)0)
#define __prof_trigger(x) ((void)0)

inline uint32_t kw_emul_byte_perm(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t bytes[8];
  bytes[0] = (a >> 0) & 0xFF;
  bytes[1] = (a >> 8) & 0xFF;
  bytes[2] = (a >> 16) & 0xFF;
  bytes[3] = (a >> 24) & 0xFF;
  bytes[4] = (b >> 0) & 0xFF;
  bytes[5] = (b >> 8) & 0xFF;
  bytes[6] = (b >> 16) & 0xFF;
  bytes[7] = (b >> 24) & 0xFF;
  uint32_t r = 0;
  for (int i = 0; i < 4; i++) {
    uint32_t idx = (sel >> (4 * i)) & 0xF;
    if (idx > 7) idx = 0;
    r |= bytes[idx] << (8 * i);
  }
  return r;
}

inline unsigned int atomicAdd(unsigned int *p, unsigned int v) {
  unsigned int old = *p;
  *p = old + v;
  return old;
}
inline unsigned long long atomicAdd(unsigned long long *p, unsigned long long v) {
  unsigned long long old = *p;
  *p = old + v;
  return old;
}
inline unsigned int atomicOr(unsigned int *p, unsigned int v) {
  unsigned int old = *p;
  *p = old | v;
  return old;
}
inline int atomicExch(int *p, int v) {
  int old = *p;
  *p = v;
  return old;
}

// ---------------------------------------------------------------------------------
// 64 bit add / sub / mul with carry emulation (bit exact w.r.t. the PTX macros)
//
// The PTX macros of GPUMath.h communicate through the hardware carry flag.
// Here the carry flag is a plain variable. PTX semantics used by GPUMath.h:
//
//   UADDO  c = a + b        carry out   -> flag
//   UADDC  c = a + b + flag carry out   -> flag
//   UADD   c = a + b + flag (flag untouched)
//   USUBO  c = a - b        borrow out  -> flag
//   USUBC  c = a - b - flag borrow out  -> flag
//   USUB   c = a - b - flag (flag untouched)
//   UMULLO lo(a*b) / UMULHI hi(a*b)
//   MADDO  r = hi(a*b) + c + flag, carry out -> flag
//   MADDC  r = hi(a*b) + c + flag, carry out -> flag
//   MADD   r = hi(a*b) + c + flag (flag untouched)
//   MADDS  same as MADD (two's complement high word is sign independent)
// ---------------------------------------------------------------------------------

static volatile uint64_t kw_emul_carry = 0;

#define KW_EMU_ADD128(res, x, y) ((unsigned __int128)(res) = (unsigned __int128)(x) + (unsigned __int128)(y))

#define UADDO(c, a, b) do { \
  unsigned __int128 _t = (unsigned __int128)(a) + (unsigned __int128)(b); \
  (c) = (uint64_t)_t; kw_emul_carry = (uint64_t)(_t >> 64); } while (0)

#define UADDC(c, a, b) do { \
  unsigned __int128 _t = (unsigned __int128)(a) + (unsigned __int128)(b) + kw_emul_carry; \
  (c) = (uint64_t)_t; kw_emul_carry = (uint64_t)(_t >> 64); } while (0)

#define UADD(c, a, b) do { \
  unsigned __int128 _t = (unsigned __int128)(a) + (unsigned __int128)(b) + kw_emul_carry; \
  (c) = (uint64_t)_t; } while (0)

#define UADDO1(c, a) UADDO(c, (c), (a))
#define UADDC1(c, a) UADDC(c, (c), (a))
#define UADD1(c, a)  UADD(c, (c), (a))

#define USUBO(c, a, b) do { \
  uint64_t _a = (a), _b = (b); \
  (c) = _a - _b; kw_emul_carry = (_a < _b) ? 1ULL : 0ULL; } while (0)

#define USUBC(c, a, b) do { \
  unsigned __int128 _t = (unsigned __int128)(a) - (unsigned __int128)(b) - kw_emul_carry; \
  (c) = (uint64_t)_t; kw_emul_carry = (_t >> 64) ? 1ULL : 0ULL; } while (0)

#define USUB(c, a, b) do { \
  unsigned __int128 _t = (unsigned __int128)(a) - (unsigned __int128)(b) - kw_emul_carry; \
  (c) = (uint64_t)_t; } while (0)

#define USUBO1(c, a) USUBO(c, (c), (a))
#define USUBC1(c, a) USUBC(c, (c), (a))
#define USUB1(c, a)  USUB(c, (c), (a))

#define UMULLO(lo, a, b) do { \
  (lo) = (uint64_t)((unsigned __int128)(a) * (unsigned __int128)(b)); } while (0)

#define UMULHI(hi, a, b) do { \
  (hi) = (uint64_t)(((unsigned __int128)(a) * (unsigned __int128)(b)) >> 64); } while (0)

// mad.hi.cc / madc.hi (PTX ISA 8.x):
//   mad.hi.cc.u64  d, a, b, c  ->  d = hi(a*b) + c        carry-out written,
//                                                         carry-in NOT consumed
//   madc.hi.cc.u64 d, a, b, c  ->  d = hi(a*b) + c + cin  carry-out written
//   madc.hi.u64    d, a, b, c  ->  d = hi(a*b) + c + cin  carry-out untouched
//   madc.hi.s64    d, a, b, c  ->  same, signed high word (identical bits)
#define MADDO(r, a, b, c) do { \
  unsigned __int128 _p = (unsigned __int128)(a) * (unsigned __int128)(b); \
  unsigned __int128 _t = (_p >> 64) + (unsigned __int128)(c); \
  (r) = (uint64_t)_t; kw_emul_carry = (uint64_t)(_t >> 64); } while (0)

#define MADDC(r, a, b, c) do { \
  unsigned __int128 _p = (unsigned __int128)(a) * (unsigned __int128)(b); \
  unsigned __int128 _t = (_p >> 64) + (unsigned __int128)(c) + kw_emul_carry; \
  (r) = (uint64_t)_t; kw_emul_carry = (uint64_t)(_t >> 64); } while (0)

#define MADD(r, a, b, c) do { \
  unsigned __int128 _p = (unsigned __int128)(a) * (unsigned __int128)(b); \
  unsigned __int128 _t = (_p >> 64) + (unsigned __int128)(c) + kw_emul_carry; \
  (r) = (uint64_t)_t; } while (0)

// madc.hi.s64: signed high word. The 128 bit product hi word is identical for
// the signed and unsigned interpretations of the operands, but the "hi"
// variant of a signed 64x64 multiply must be computed from the signed
// product, i.e. a\*b >> 64 with a and b taken as int64_t.
#define MADDS(r, a, b, c) do { \
  __int128 _p = (__int128)(int64_t)(a) * (__int128)(int64_t)(b); \
  __int128 _t = (_p >> 64) + (__int128)(int64_t)(c) + (__int128)kw_emul_carry; \
  (r) = (uint64_t)_t; } while (0)

// ---------------------------------------------------------------------------------
// Kernel launch emulation
//
// KW_LAUNCH(kernel, grid, block, args...) runs <kernel> for every (block,
// thread) pair of the requested launch geometry, with threadIdx/blockIdx/
// blockDim set accordingly. Threads are executed one after the other.
// ---------------------------------------------------------------------------------

template <typename K, typename... A>
void kwEmulateKernel(int grid, int block, K kernel, A... args) {

  kw_emul_gridDim.x = (unsigned int)grid;
  kw_emul_gridDim.y = 1;
  kw_emul_gridDim.z = 1;
  kw_emul_blockDim.x = (unsigned int)block;
  kw_emul_blockDim.y = 1;
  kw_emul_blockDim.z = 1;

  for (int b = 0; b < grid; b++) {
    kw_emul_blockIdx.x = (unsigned int)b;
    kw_emul_blockIdx.y = 0;
    kw_emul_blockIdx.z = 0;
    for (int t = 0; t < block; t++) {
      kw_emul_threadIdx.x = (unsigned int)t;
      kw_emul_threadIdx.y = 0;
      kw_emul_threadIdx.z = 0;
      kw_emul_carry = 0;
      kernel(args...);
    }
  }

}

#define KW_LAUNCH(kernel, grid, block, ...) kwEmulateKernel((int)(grid), (int)(block), kernel, __VA_ARGS__)

#endif // KW_HAVE_PTX_ASM / KW_EMULATE

#if KW_HAVE_PTX_ASM
#define KW_LAUNCH(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)
#endif

#endif // GPUCOMPAT_H
