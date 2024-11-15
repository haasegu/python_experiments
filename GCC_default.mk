# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=GCC_

CC	= gcc
CXX     = g++
F77	= gfortran
LINKER  = ${CXX}

#LINKFLAGS   += -lblas
# The <cblas.h> header requires extern "C".

WARNINGS = -Wall -pedantic -Wextra -Weffc++ -Woverloaded-virtual -Wfloat-equal -Wshadow \
           -Wredundant-decls -fmax-errors=1
# -Winline
# -Wunreachable-code
#CXXFLAGS += -ffast-math -march=native ${WARNINGS}
CXXFLAGS += -O3 -std=c++20 ${WARNINGS}
# -funroll-all-loops
#-msse3
# -ftree-vectorizer-verbose=2  -DNDEBUG
# -ftree-vectorizer-verbose=5
# -ftree-vectorize -fdump-tree-vect-blocks=foo.dump  -fdump-tree-pre=stderr

# CFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp -fdump-tree-vect-details
# CFLAGS	= -ffast-math -O3 -funroll-loops -DNDEBUG -msse3 -fopenmp -ftree-vectorizer-verbose=2
# #CFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
# FFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
# LFLAGS  = -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
LINKFLAGS   += -O3

#architecture
#CPU = -march=znver2
CXXFLAGS  += ${CPU}
LINKFLAGS += ${CPU}

# different libraries in Ubuntu or manajaró
ifndef UBUNTU
UBUNTU=1
endif

# BLAS, LAPACK
ifeq ($(UBUNTU),1)
LINKFLAGS += -llapack -lblas
# -lopenblas
else
# on  archlinux
LINKFLAGS += -llapack -lopenblas -lcblas
endif

# interprocedural optimization
#CXXFLAGS  += -flto
#LINKFLAGS += -flto

# for debugging purpose (save code)
# -fsanitize=leak         # only one out the three can be used
# -fsanitize=address
# -fsanitize=thread
SANITARY =  -fsanitize=address  -fsanitize=undefined -fsanitize=null -fsanitize=return \
 -fsanitize=bounds -fsanitize=alignment -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow \
 -fsanitize=bool -fsanitize=enum -fsanitize=vptr
#CXXFLAGS  += ${SANITARY}
#LINKFLAGS += ${SANITARY}

# profiling tools
#CXXFLAGS  += -pg
#LINKFLAGS += -pg

default: ${PROGRAM}

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean:
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	-@rm -f *_ *~ *.bak *.log *.out *.tar *.orig *.optrpt
	-@rm -rf html

run: clean ${PROGRAM}
#run: ${PROGRAM}
#	time  ./${PROGRAM} ${PARAMS}
	./${PROGRAM} ${PARAMS}

# tar the current directory
MY_DIR = `basename ${PWD}`
tar: clean_all
	@echo "Tar the directory: " ${MY_DIR}
	@cd .. ;\
	tar cf ${MY_DIR}.tar ${MY_DIR} *default.mk ;\
	cd ${MY_DIR}
# 	tar cf `basename ${PWD}`.tar *

zip: clean
	@echo "Zip the directory: " ${MY_DIR}
	@cd .. ;\
	rm -f ${MY_DIR}.zip; \
	zip -r ${MY_DIR}.zip ${MY_DIR} *default.mk ;\
	cd ${MY_DIR}

doc:
	doxygen Doxyfile

#########################################################################
.SUFFIXES: .f90

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<
#	$(CXX) -c $(CXXFLAGS) -o $@ $<  2>&1 | tee -a $<.log 
#	$(CXX) -c $(CXXFLAGS) -o $@ $<  2>&1 | tee -a $(<:.cpp=.log)

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f.o:
	$(F77) -c $(FFLAGS) -o $@ $<

.f90.o:
	$(F77) -c $(FFLAGS) -o $@ $<

##################################################################################################
#    some tools
# Cache behaviour (CXXFLAGS += -g  tracks down to source lines; no -pg in linkflags)
cache: ${PROGRAM}
	valgrind --tool=callgrind --simulate-cache=yes ./$^  ${PARAMS}
#	kcachegrind callgrind.out.<pid> &
	kcachegrind `ls -1tr  callgrind.out.* |tail -1`

# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
# no "-pg"  in compile/link options
mem: ${PROGRAM}
	valgrind -v --leak-check=yes --tool=memcheck --undef-value-errors=yes --track-origins=yes --log-file=$^.addr.out --show-reachable=yes ./$^  ${PARAMS}
# Graphical interface
# valkyrie

#  Simple run time profiling of your code
#  CXXFLAGS += -g -pg
#  LINKFLAGS += -pg
prof: ${PROGRAM}
	./$^  ${PARAMS}
	gprof -b ./$^ > gp.out
#	kprof -f gp.out -p gprof &

#                   sudo apt install gprofng-gui
#                   https://parallel.computer/presentations/PPoPP2023/2023-Ruud-Slides.pdf
#   read §3 in      https://sourceware.org/binutils/docs/gprofng.html
# /usr/bin/gp-collect-app -o test.1.er -p on -S on /home/ghaase/Lectures/Math2CPP/Codes/seq/jacobi_oo_stl/main.GCC_ 
prof2: ${PROGRAM}
	gprofng collect app -h auto ./$^  ${PARAMS}
#	gprofng display text -functions `ls -1tdr test.*.er |tail -1`
	gprofng display text -script gprofng_script2 `ls -1tdr test.*.er |tail -1`
#	gprofng display text -script gprofng_script2 test.*.er
# https://blogs.oracle.com/linux/post/the-gprofng-gui
	gprofng display gui &
	
prof3: ${PROGRAM}
	perf record ./$^  ${PARAMS}
	perf report
#  perf in Ubuntu 20.04:   https://www.howtoforge.com/how-to-install-perf-performance-analysis-tool-on-ubuntu-20-04/
#  * install 
#  * sudo vi /etc/sysctl.conf
#                add   kernel.perf_event_paranoid = 0

#Trace your heap:
#> heaptrack ./main.GCC_
#> heaptrack_gui heaptrack.main.GCC_.<pid>.gz
heap: ${PROGRAM}
	heaptrack ./$^  ${PARAMS} 11
	heaptrack_gui  `ls -1tr  heaptrack.$^.* |tail -1` &

# misra: see https://www.synopsys.com/software-integrity/static-analysis-tools-sast/misra.html
codecheck: $(SOURCES)
	cppcheck --enable=all --inconclusive --std=c++20 --suppress=missingIncludeSystem --checkers-report=chk.txt $^

#	cppcheck --enable=all --inconclusive --std=c++20 --addon=misra.py --suppress=missingIncludeSystem $^ --xml 2> err.xml 
#	cppcheck-htmlreport --file=err.xml --report-dir=test1 --source-dir=.

########################################################################
#  get the detailed  status of all optimization flags
info:
	echo "detailed  status of all optimization flags"
	$(CXX) --version
	$(CXX) -Q $(CXXFLAGS) --help=optimizers
	lscpu
	inxi -C
	lstopo

# Excellent hardware info
#	hardinfo
# Life monitoring of CPU frequency etc.
#	sudo i7z

# Memory  consumption
#	vmstat -at -SM 3
#	xfce4-taskmanager


# https://www.tecmint.com/check-linux-cpu-information/
#https://www.tecmint.com/monitor-cpu-and-gpu-temperature-in-ubuntu/

# Debugging:
# https://wiki.archlinux.org/index.php/Debugging
