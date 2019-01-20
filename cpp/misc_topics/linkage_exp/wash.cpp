// https://www.geeksforgeeks.org/internal-linkage-external-linkage-c/
// C code to illustrate Internal Linkage 
#include <stdio.h> 
#include "animal.cpp" // note that animal is included. 
  
int main() 
{ 
    call_me(); 
    printf("\n having fun washing!"); 
    animals = 10; 
    printf("%d\n", animals); 
    return 0; 
} 
