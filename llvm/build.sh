clang -O3 -emit-llvm hello.c -c
llvm-dis < hello.bc
lli hello.bc
llc hello.bc -o hello.s
clang hello.s -o hello.native 
clang hello.c -o hello.direct
echo hello.native md5sum: 
md5sum hello.native
echo hello.direct md5sum:
md5sum hello.direct 

