#include <stdio.h>
enum amdgv_reset_mode {
AMDGV_RESET_BEGIN,
AMDGV_RESET_MODE1 = 1,
AMDGV_RESET_MODE2,
AMDGV_RESET_MODE2_BACO,
AMDGV_RESET_BACO,
AMDGV_RESET_PF_FLR,
AMDGV_RESET_END,
};

main()
{
    int i ;
    enum amdgv_reset_mode test;
    for (i = AMDGV_RESET_BEGIN; i <= AMDGV_RESET_END; i++) {
        printf("%x.\n", i);
    }

    printf("Done.\n");
}
