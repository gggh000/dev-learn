kernel void kernelfcn(global uint * c, global uint * b,global uint * a) {
    uint gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
