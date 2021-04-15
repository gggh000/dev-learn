mkdir build ; cd build
cmake -DCMAKE_FIND_LIBRARY_PREFIXES=lib -DCMAKE_FIND_LIBRARY_SUFFIXES=.so ..
make p41
make p25-cuda
make p189
make ex-code-1
