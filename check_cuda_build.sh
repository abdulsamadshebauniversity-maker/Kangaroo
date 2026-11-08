#!/bin/bash
# ---------------------------------------------------------------------------------
# check_cuda_build.sh - first thing to run on a CUDA machine / Google Colab.
#
# It answers the only question that could not be answered in the development
# environment (no GPU, no CUDA toolkit there):
#
#     does the CUDA build of the GPU Xpoint / BSGS modes actually compile and
#     do the kernels produce the right answers on real hardware?
#
# Usage:  ./check_cuda_build.sh [compute_capability]
#         ./check_cuda_build.sh 75     # T4 (Colab free)
# ---------------------------------------------------------------------------------

set -e

CCAP=${1:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')}
if [ -z "$CCAP" ]; then
  echo "No GPU detected, using compute capability 75 (T4)"
  CCAP=75
fi

echo "=================================================================="
echo " 1. environment"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || {
  echo "nvidia-smi not found: this is not a CUDA machine."; exit 1; }
echo
nvcc --version | tail -2 || { echo "nvcc not found: install the CUDA toolkit first."; exit 1; }
echo
echo "CUDA_HOME = ${CUDA_HOME:-<not set>}"
echo "building for compute capability $CCAP"
echo

echo "=================================================================="
echo " 2. build (GPU build: nvcc compiles the kernels and the drivers)"
echo "=================================================================="
make clean >/dev/null 2>&1 || true
if ! make gpu=1 ccap=$CCAP 2>&1 | tee /tmp/build.log; then
  echo
  echo "BUILD FAILED - the compile errors above are exactly what has to be"
  echo "fixed in the sources. The test suite (make check) does not need CUDA,"
  echo "so it still passes; only the nvcc path is affected."
  exit 1
fi
echo
echo "build OK -> ./kangaroo"
grep -i "registers\|spill" /tmp/build.log | head -20 || true
echo

echo "=================================================================="
echo " 3. correctness on the emulator (no GPU used, bit-exact SIMT)"
echo "=================================================================="
make check 2>&1 | grep -E "checks|FAIL|found|Running"

echo
echo "=================================================================="
echo " 4. correctness on the real device"
echo "=================================================================="
# The command line front end uses the same drivers the GPU build links.
KEY_X=8208F5ABF04066BAD1DB9D46F8BCF5A6CC11D0558AB523E7BD3C0EC08BDB782F
KEY_PUB=048208F5ABF04066BAD1DB9D46F8BCF5A6CC11D0558AB523E7BD3C0EC08BDB782FB7A0AC7E4A033B943B42175CA60CB78F65BDACE71333FF53E12E50900800D4DA

echo "--- xpoint (key 0x12D687 must be found)"
./kangaroo --mode xpoint --start 0x120000 --range 200000 --x $KEY_X --quiet
echo "--- bsgs   (key 0x12D687 must be found)"
./kangaroo --mode bsgs --start 0x120000 --range 200000 --pubkey $KEY_PUB --m 4096 --quiet

echo
echo "=================================================================="
echo " 5. throughput"
echo "=================================================================="
make bench 2>&1 | tail -3

echo
echo "=================================================================="
echo " done. If every step printed a key/s line, the GPU port works."
echo " Copy the 'registers' lines of step 2 and the step 5 numbers back."
echo "=================================================================="
