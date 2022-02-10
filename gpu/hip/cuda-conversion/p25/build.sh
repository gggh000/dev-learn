rm -rf p25-*
rm -rf offload*
rm -rf *.o
rm -rf *.bc

ARCH=gfx90a
ARCH2=gfx908
hipcc -c -emit-llvm --cuda-device-only --offload-arch=$ARCH --offload-arch=$ARCH2 p25.cpp
hipcc -c -emit-llvm --cuda-host-only -target x86_64-linux-gnu -o p25.bc p25.cpp
hipcc -c p25.bc -o p25.o

clang -target amdgcn-amd-amdhsa -mcpu=$ARCH p25-hip-amdgcn-amd-amdhsa-$ARCH.bc -o p25-hip-amdgcn-amd-amdhsa-$ARCH.o
clang -target amdgcn-amd-amdhsa -mcpu=$ARCH2 p25-hip-amdgcn-amd-amdhsa-$ARCH2.bc -o p25-hip-amdgcn-amd-amdhsa-$ARCH2.o
clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-$ARCH,hip-amdgcn-amd-amdhsa-$ARCH2 -inputs=/dev/null,p25-hip-amdgcn-amd-amdhsa-$ARCH.o,p25-hip-amdgcn-amd-amdhsa-$ARCH2.o -outputs=offload_bundle.hipfb

llvm-mc hip_obj_gen.mcin -o p25-device.o --filetype=obj
hipcc p25.o p25-device.o -o a.out
echo "clang offload bundler --list:"
clang-offload-bundler --list --inputs=offload_bundle.hipfb  --type=o

