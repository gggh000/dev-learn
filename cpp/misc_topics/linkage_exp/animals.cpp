// https://www.geeksforgeeks.org/internal-linkage-external-linkage-c/
// C code to illustrate Internal Linkage 
// This code implements static linkage on identifier animals

#include <stdio.h> 
#include "animals.h"
  
static int animals = 8; 
const int i = 5; 
  
int call_me(void) { 
    printf("%d %d", i, animals); 
}
