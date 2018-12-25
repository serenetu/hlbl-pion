SHELL=/bin/bash

CC=mpicc -fopenmp -O2 -w
CXX=mpic++ -std=c++0x -fopenmp -O2 -w

qlat_prefix=$$HOME/qlat-build/1.0

QLAT_INCLUDE=$(qlat_prefix)/include
QLAT_LIB=$(qlat_prefix)/lib

QLAT_CFLAGS=-fno-strict-aliasing -DEIGEN_DONT_ALIGN_STATICALLY
QLAT_CFLAGS+= -I$(QLAT_INCLUDE)
QLAT_CFLAGS+= -I$(QLAT_INCLUDE)/eigen3
QLAT_CFLAGS+= -I$(QLAT_INCLUDE)/utils
QLAT_CFLAGS+= -I./cuba-cooley

QLAT_CXXFLAGS=$(QLAT_CFLAGS)

QLAT_LDFLAGS=-L$(QLAT_LIB)
QLAT_LDFLAGS+= -lgsl -lgslcblas -lm
QLAT_LDFLAGS+= -lfftw3_omp -lfftw3
QLAT_LDFLAGS+= -lz
QLAT_LDFLAGS+= -L./cuba-cooley -lcuba

all: hlbl-pi.x

run: hlbl-pi.x
	. $(qlat_prefix)/setenv.sh ; time mpirun --np 1 ./hlbl-pi.x
	make clean

hlbl-pi.x: *.C
	. $(qlat_prefix)/setenv.sh ; time make build
	[ -f $@ ]

build:
	-$(CXX) -o hlbl-pi.x $(QLAT_CXXFLAGS) *.C $(QLAT_LDFLAGS) 2>&1 | grep --color=always 'error:\|'

clean:
	-rm hlbl-pi.x

show-info:
	@echo CXX: $(CXX)
	@echo CXXFLAGS: $(QLAT_CXXFLAGS)
	@echo LDFLAGS: $(QLAT_LDFLAGS)
#. $(qlat_prefix)/setenv.sh ; time mpirun -x OMP_NUM_THREADS=2 --np 8 ./qlat.x
