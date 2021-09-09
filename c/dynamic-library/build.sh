echo "Creating shared library."
echo "Creating  *.o files..."
gcc libfile*.c -c -fPIC
echo "Generating *.so library..."
gcc *.o -shared -o liball.so
echo "Linking and test..."
gcc -L. test.c -lall -o test-static-lib
if [[ -z $LD_LIBRARY_PATH ]]  ; then
    export LD_LIBRARY_PATH=$PWD
else
    export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
fi
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
./test-static-lib

