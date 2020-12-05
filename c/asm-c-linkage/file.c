#include <stdio.h>

int functionC() {
    int a = 0xbeef;
    return a;
    //printf("functionC called...");
}

int functionC2(int p1, int p2) {
    int c;
    int a = 0xdead;
    c =  p1 * 2;
    return c;
}
