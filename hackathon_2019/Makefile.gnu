
# Adjust these as needed
CXX = g++
CXXFLAGS = -O3 -march=native -fPIC -g -std=c++11 -fopenmp
LDFLAGS =
LINK = -fopenmp -lm

OBJS = toast_pointing.o pointing_openmp.o utils.o
HEADERS = utils.hpp pointing_openmp.hpp


all : toast_pointing

toast_pointing : $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LINK)

%.o : %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I. -o $@ -c $<

clean :
	rm -f toast_pointing *.o
