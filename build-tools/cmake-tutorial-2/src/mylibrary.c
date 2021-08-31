#include <stdio.h>

void mylibfcn1() {
    printf("mylibfcn1: entered...\n"); 
    #ifdef MYLIBDEF
        char *message2 = "MYLIBDEF is defined!\n";
    #else
        char *message2 = "MYLIBDEF is not defined!\n";
    #endif
    printf("mylibfcn1: entered... message: %s", message2);
}

int mylibfcn2() {
    printf("mylibfcn2: entered... \n");
    return 0;
}
