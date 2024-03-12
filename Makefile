LDFLAGS:=-lm $(shell pkg-config --libs opencv4 libpq)
CXXFLAGS:=-O2 -std=c++11 -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a $(shell pkg-config --cflags opencv4 libpq)
CPP=g++

all: main

%.o: %.cpp $(DEPS)
	$(CPP) -std=c++11 $(COMMON) $(CXXFLAGS) -c $< -o $@

main: main.o openface.o moving_average.o selfiesegment.o
	$(CPP) -std=c++11 $(COMMON) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm *.o main
