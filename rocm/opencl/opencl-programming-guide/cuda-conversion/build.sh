mkdir build ; cd build
rm -rf ./*
CXX=/opt/rocm-4.0.0/llvm/bin/clang cmake ..
for i in  p25 p31 p41 p188 ex-code-1 ; do
    make $i
done
