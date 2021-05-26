#include <CL/cl.h>
#include <stdio.h>

kernel void kernelfcn(      global uint *dev_c,   global uint * dev_a,  global uint * dev_b)      
{                                           
  uint tid = get__id(0);               
 *dev_c = 100;                              
 *dev_a = 200;                              
}                                           ;

