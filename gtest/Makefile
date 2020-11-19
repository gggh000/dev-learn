LDFLAGS = -lgtest -lgtest_main -lgmock -lgmock_main -pthread
GOOGLETEST=/usr/src/googletest/

all: main.c
	g++ -o main main.c -I$(GOOGLETEST)/googletest/include
	g++ -o main.o $(LDFLAGS)
