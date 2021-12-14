mkdir build ; cd build
rm -rf ./*
CXX=/opt/rocm-4.3.0/llvm/bin/clang cmake ..

for i in p46 ; do
    make $i
done

