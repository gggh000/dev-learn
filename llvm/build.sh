# generate bitcode with -emit-llvm flag.

clang -O3 -emit-llvm hello.c -c

# input: bitcode, output: llvm ir.

llvm-dis < hello.bc > hello.ll

# jit input: bitcode.

lli hello.bc

# output assembly: input: bitcode, output: assembly.

llc hello.bc -o hello.s

# compile from assembly.

clang hello.s -o hello.native 

# compile from source.

clang hello.c -o hello.direct
echo hello.native md5sum: 
md5sum hello.native
echo hello.direct md5sum:
md5sum hello.direct 

