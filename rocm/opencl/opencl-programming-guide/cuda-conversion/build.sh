mkdir build ; cd build
rm -rf ./*
cmake ..
for i in  p25 p31 p41 p188 ex-code-1 ; do
    make $i
done
