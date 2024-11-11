#
# use GNU-Compiler tools
COMPILER=GCC_
# alternatively from the shell
# export COMPILER=GCC_
# or, alternatively from the shell
# make COMPILER=GCC_

# use Intel compilers
#COMPILER=ICC_

# use PGI compilers
# COMPILER=PGI_

# use CLANG compilers
# COMPILER=CLANG_


SOURCES = main.cpp mylib.cpp
ifeq ($(COMPILER),NVCC_)
SOURCES += mylib_gpu.cu
endif
#OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS = $(addsuffix .o,  $(basename $(SOURCES)) )

PROGRAM	= main.${COMPILER}

# OpenMP  (the NVCC contains already the OpenMP flags for the host)
ifneq ($(COMPILER),NVCC_)
CXXFLAGS += -fopenmp
LINKFLAGS += -fopenmp
endif
# uncomment the next to lines for debugging and detailed performance analysis
CXXFLAGS += -g -pg
LINKFLAGS += -g -pg
# do not use -pg with PGI compilers

ifndef COMPILER
  COMPILER=GCC_
endif

include ../${COMPILER}default.mk


task:
	@pdflatex task
	@pdflatex task

