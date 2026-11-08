#---------------------------------------------------------------------
# Makefile for BSGS
#
# Author : Jean-Luc PONS

ifdef gpu

SRC = SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      GPU/GPUEngine.o GPU/GPULaunch.o GPU/GPUBsgs.o GPU/GPUXPoint.o \
      Kangaroo.cpp HashTable.cpp Backup.cpp Thread.cpp Check.cpp Network.cpp \
      Merge.cpp PartMerge.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      GPU/GPUEngine.o GPU/GPULaunch.o GPU/GPUBsgs.o GPU/GPUXPoint.o \
      Kangaroo.o HashTable.o Backup.o Check.o Network.o Merge.o PartMerge.o)

else

SRC = SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      Kangaroo.cpp HashTable.cpp Thread.cpp Check.cpp \
      Backup.cpp Network.cpp Merge.cpp PartMerge.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      Kangaroo.o HashTable.o Thread.o Check.o Backup.o \
      Network.o Merge.o PartMerge.o)

endif

CXX        = g++
# Toolchain is autodetected ($(CUDA) and $(CXXCUDA) can be overridden on the
# command line). The old hardcoded cuda-8.0 / g++-4.8 pair does not exist on
# current distributions and cannot compile the C++11 headers of the project.
CUDA      ?= $(shell if [ -d /usr/local/cuda ]; then echo /usr/local/cuda; \
              elif [ -n "$$CUDA_HOME" ]; then echo $$CUDA_HOME; \
              else dirname $$(dirname $$(which nvcc 2>/dev/null)) 2>/dev/null; fi)
NVCC      ?= $(if $(CUDA),$(CUDA)/bin/nvcc,/usr/bin/nvcc)
CXXCUDA   ?= $(CXX)
# GPU compute capability for -gencode (default 61 = Pascal, override with
# "make gpu=1 ccap=86"). Use 'ccap' in rules: make variables are case sensitive
# and the old file mixed both spellings.
ccap      ?= 61

ifdef gpu

ifdef debug
CXXFLAGS   = -DWITHGPU -m64  -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart

else

ifdef debug
CXXFLAGS   = -m64 -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   =  -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread

endif

#--------------------------------------------------------------------

ifdef gpu
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu | $(OBJDIR)/GPU
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu | $(OBJDIR)/GPU
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif
endif

$(OBJDIR)/GPU/GPULaunch.o: GPU/GPULaunch.cu GPU/GPULaunch.h GPU/GPUXPointKernel.h GPU/GPUBsgsKernel.h | $(OBJDIR)/GPU
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC \
	  -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -I. \
	  -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPULaunch.o -c GPU/GPULaunch.cu

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: bsgs

bsgs: $(OBJET)
	@echo Making Kangaroo...
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo

$(OBJET): | $(OBJDIR) $(OBJDIR)/SECPK1 $(OBJDIR)/GPU

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

$(OBJDIR)/SECPK1: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p SECPK1

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/SECPK1/*.o


#---------------------------------------------------------------------
# CPU-side test build of the GPU device code (no GPU required)
#
#   make check-arithmetic
#
# Compiles GPU/GPUMath.h through the SIMT emulation layer of
# GPU/GPUCompat.h (-DKW_EMULATE) and validates every 256 bit primitive
# against the independent fixed size CPU implementation in SECPK1/.
#---------------------------------------------------------------------

TESTDIR   = obj/test
TESTCORE  = SECPK1/Int.cpp SECPK1/IntMod.cpp SECPK1/IntGroup.cpp \
            SECPK1/SECP256K1.cpp SECPK1/Point.cpp SECPK1/Random.cpp Timer.cpp
TESTFLAGS = -O2 -m64 -mssse3 -DKW_EMULATE -Wno-write-strings -I.

check-pointwalk: $(TESTDIR)/test_pointwalk
	@echo "Running GPU group walk tests..."
	@$(TESTDIR)/test_pointwalk 5

$(TESTDIR)/test_pointwalk: tests/test_pointwalk.cpp GPU/GPUPointWalk.h GPU/GPUMath.h GPU/GPUCompat.h $(TESTCORE)
	@mkdir -p $(TESTDIR)
	$(CXX) $(TESTFLAGS) -o $@ tests/test_pointwalk.cpp $(TESTCORE)

check-bsgs: $(TESTDIR)/test_bsgs
	@echo "Running GPU BSGS engine tests..."
	@$(TESTDIR)/test_bsgs

$(TESTDIR)/test_bsgs: tests/test_bsgs.cpp GPU/GPUBsgs.cpp GPU/GPUBsgs.h GPU/GPUBsgsKernel.h GPU/GPUPointWalk.h $(TESTCORE)
	@mkdir -p $(TESTDIR)
	$(CXX) $(TESTFLAGS) -o $@ tests/test_bsgs.cpp GPU/GPUBsgs.cpp $(TESTCORE)

check-xpoint: $(TESTDIR)/test_xpoint
	@echo "Running GPU xpoint kernel tests..."
	@$(TESTDIR)/test_xpoint

$(TESTDIR)/test_xpoint: tests/test_xpoint.cpp GPU/GPUXPoint.cpp GPU/GPUXPoint.h GPU/GPUXPointKernel.h GPU/GPUPointWalk.h $(TESTCORE)
	@mkdir -p $(TESTDIR)
	$(CXX) $(TESTFLAGS) -o $@ tests/test_xpoint.cpp GPU/GPUXPoint.cpp $(TESTCORE)

bench: $(TESTDIR)/bench_gpu
	@$(TESTDIR)/bench_gpu

$(TESTDIR)/bench_gpu: tests/bench_gpu.cpp GPU/GPUBsgs.cpp GPU/GPUXPoint.cpp GPU/GPUBsgs.h GPU/GPUXPoint.h $(TESTCORE)
	@mkdir -p $(TESTDIR)
	$(CXX) $(TESTFLAGS) -o $@ tests/bench_gpu.cpp GPU/GPUBsgs.cpp GPU/GPUXPoint.cpp $(TESTCORE)

check: check-arithmetic check-pointwalk check-xpoint check-bsgs

check-arithmetic: $(TESTDIR)/test_gpumath
	@echo "Running device arithmetic tests..."
	@$(TESTDIR)/test_gpumath 64

$(TESTDIR)/test_gpumath: tests/test_gpumath.cpp GPU/GPUMath.h GPU/GPUCompat.h $(TESTCORE)
	@mkdir -p $(TESTDIR)
	$(CXX) $(TESTFLAGS) -o $@ tests/test_gpumath.cpp $(TESTCORE)
