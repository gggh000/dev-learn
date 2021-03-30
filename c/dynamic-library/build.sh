echo Creating  *.o files...
gcc lib*.c -c -fPIC

echo Generating *.so library...
gcc *.o -shared -o liball.so

echo Linking and test...
gcc -L`pwd` test.c -lall -o test

export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
