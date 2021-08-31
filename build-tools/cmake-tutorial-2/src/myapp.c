#include <stdio.h>
#include "mylibrary.h"

int main()
{
    #ifdef AWESOME
        char *message = "YOU ARE AWESOME!\n";
    #else
        char *message = "Sorry, you are not awesome\n";
    #endif
    printf("%s", message);

    printf("Calling mylibfcn2...\n");
    mylibfcn1();
    printf("Back from mylibfcn2...\n");
    return 0;
}
