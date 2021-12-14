#include <stdio.h>
typedef enum months {JAN=100, FEB, MAR=100};

int main()
{

    int i;
    printf("Jan: 0x%x\n", JAN);
    printf("Feb: 0x%x\n", FEB);
    printf("Mar: 0x%x\n", MAR);
    printf("\n");
}
