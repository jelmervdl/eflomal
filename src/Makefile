# CFLAGS=-Ofast -march=native -Wall --std=c++17 -Wno-unused-function -g
# This is more suitable for debugging:
CFLAGS=-Wall --std=gnu99 -Wno-unused-function -g
CCFLAGS=-Wall --std=c++17 -Wno-unused-function -g
LDFLAGS=-lm
# -lrt
INSTALLDIR=/usr/local/bin
PYINSTALLDIR=../python/eflomal/bin

all: eflomal eflomal++

eflomal.o: eflomal.c natmap.c hash.c random.c simd_math_prims.h
	$(CC) $(CFLAGS) -c eflomal.c

eflomal: eflomal.o

eflomal++.o: eflomal.cc random.c simd_math_prims.h
	$(CC) $(CCFLAGS) -o eflomal++.o -c eflomal.cc

eflomal++: eflomal++.o
	$(CC) $(LDFLAGS) -lstdc++ -o eflomal++ -lstdc++ eflomal++.o

python-install: eflomal
	mkdir -p $(PYINSTALLDIR)
	cp eflomal $(PYINSTALLDIR)

install: eflomal
	install -t $(INSTALLDIR) eflomal

clean:
	rm -f eflomal eflomal.o eflomal++ eflomal++.o

