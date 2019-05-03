#
# You should define the $CONFIG environment variable, either by exporting it or
# simply prefixing it to the make command like this:
#
#  %>  CONFIG=cuda-61 make clean
#  %>  CONFIG=cuda-61 make
#
# Then create / edit the compiler definitions in the corresponding file in the
# config subdirectory.
#

ifndef CONFIG
  CONFIG := $(error CONFIG undefined)undefined
endif
include config/$(CONFIG)

# Make seems to delete these sometimes...
.PRECIOUS : $(wildcard config/*)

HEAD_COMMON = \
utils.hpp

HEAD_OMP = \
pointing_openmp.hpp \
solver_openmp.hpp

HEAD_CUDA = \
pointing_cuda.hpp \
solver_cuda.hpp

OBJS_COMMON = \
utils.o

OBJS_OMP = \
pointing_openmp.o \
solver_openmp.o

OBJS_CUDA = \
pointing_cuda.o \
solver_cuda.o

LIB_COMMON = libtoastkernel_common.a
LIB_OMP = libtoastkernel_omp.a
LIB_CUDA = libtoastkernel_cuda.a

PNTG_OBJS = \
toast_pointing_omp.o \
toast_pointing_cuda.o

PNTG_PROGS = \
toast_pointing_omp \
toast_pointing_cuda

SOLVER_OBJS = \
toast_solver_omp.o \
toast_solver_cuda.o

SOLVER_PROGS = \
toast_solver_omp \
toast_solver_cuda


all : $(PNTG_PROGS) $(SOLVER_PROGS)

toast_pointing_omp: % : %.o $(LIB_OMP) $(LIB_COMMON)
	$(CXX) -o $@ $^ $(LIB_OMP) $(LIB_COMMON) $(OMPFLAGS) $(LINK)

toast_pointing_cuda: % : %.o $(LIB_CUDA) $(LIB_COMMON)
	$(CXX) -o $@ $^ $(LIB_CUDA) $(LIB_COMMON) $(NVLINK)

toast_solver_omp: % : %.o $(LIB_OMP) $(LIB_COMMON)
	$(CXX) -o $@ $^ $(LIB_OMP) $(LIB_COMMON) $(OMPFLAGS) $(LINK)

toast_solver_cuda: % : %.o $(LIB_CUDA) $(LIB_COMMON)
	$(CXX) -o $@ $^ $(LIB_CUDA) $(LIB_COMMON) $(NVLINK)


$(PNTG_OBJS) : $(subst _cuda.o,.cpp,$(subst _omp.o,.cpp,$@)) $(HEAD_COMMON) $(HEAD_OMP) $(HEAD_CUDA)
	$(CXX) $(CXXFLAGS) -I. -o $@ -c $(subst _cuda.o,.cpp,$(subst _omp.o,.cpp,$@))

$(SOLVER_OBJS) : $(subst _cuda.o,.cpp,$(subst _omp.o,.cpp,$@)) $(HEAD_COMMON) $(HEAD_OMP) $(HEAD_CUDA)
	$(CXX) $(CXXFLAGS) -I. -o $@ -c $(subst _cuda.o,.cpp,$(subst _omp.o,.cpp,$@))


$(LIB_COMMON) : $(OBJS_COMMON)
	ar rcs $@ $(OBJS_COMMON)

$(LIB_OMP) : $(OBJS_OMP)
	ar rcs $@ $(OBJS_OMP)

$(LIB_CUDA) : $(OBJS_CUDA)
	ar rcs $@ $(OBJS_CUDA)

$(OBJS_COMMON) : $(subst .o,.cpp,$@) $(HEAD_COMMON)
	$(CXX) $(CXXFLAGS) -I. -o $@ -c $(subst .o,.cpp,$@)

$(OBJS_OMP) : $(subst .o,.cpp,$@) $(HEAD_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I. -o $@ -c $(subst .o,.cpp,$@)

$(OBJS_CUDA) : $(subst .o,.cu,$@) $(HEAD_CUDA)
	$(NVCC) $(NVFLAGS) -I. -o $@ -c $(subst .o,.cu,$@)

clean :
	rm -f $(PNTG_PROGS) $(SOLVER_PROGS) *.o
