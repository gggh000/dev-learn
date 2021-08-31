#include <stdio.h>
#include "mylibrary.h"

void mymainfcn1() {
    printf("mylibfcn1: entered...\n");

    #ifdef MYLIBDEF
        char *message2 = "MYLIBDEF is defined in myapp.c!\n";
    #else
        char *message2 = "MYLIBDEF is not defined in myapp.c!\n";
    #endif
    printf("mymainfcn1: entered... message: %s", message2);
}

int main()
{
    #ifdef AWESOME
        char *message = "YOU ARE AWESOME!\n";
    #else
        char *message = "Sorry, you are not awesome\n";
    #endif
    printf("%s", message);

    mylibfcn1();
    mymainfcn1();
    return 0;
}
