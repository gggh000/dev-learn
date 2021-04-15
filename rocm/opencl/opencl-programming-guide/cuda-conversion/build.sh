mkdir build ; cd build
cmake \
	-DCMAKE_FIND_LIBRARY_PREFIXES=lib \
	-DCMAKE_FIND_LIBRARY_SUFFIXES=.so \
	-DZLIB_INCLUDE_DIR=/usr/include \
	-DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/ \
	.. 
make p41
make p25-cuda
make p189
make ex-code-1
