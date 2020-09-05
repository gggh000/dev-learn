#include "stdio.h"

int main() {
    struct s {
            int i;
            char m1;
            char m2;
            char m3;
            char m4;
    };

    /* This will print 1 */
    printf("%d\n", &((struct s*)0)->m1);
    printf("%d\n", &((struct s*)0)->m3);
    printf("%d\n", &((struct s*)0)->m2);
    printf("%d\n", &((struct s*)0)->m4);
}
