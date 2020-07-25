kernel void kernelfcn(     global uint *dev_c, global uint *dev_a, global uint *dev_b)  
{                                                                      
    uint tid = get_global_id(0);                                          
    dev_c[tid] = dev_a[tid] + dev_b[tid];                                 
}       
