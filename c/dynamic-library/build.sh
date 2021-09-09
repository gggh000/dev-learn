set -x
echo "Creating shared library."
echo "Creating  *.o files..."
gcc libfile*.c -c -fPIC
echo "Generating *.so library..."
gcc *.o -shared -o liball.so
echo "Linking and test..."
gcc -L. test.c -lall -o test-dynamic-lib
if [[ -z $LD_LIBRARY_PATH ]]  ; then
    export LD_LIBRARY_PATH=$PWD
else
    export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
fi
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
./test-dynamic-lib

echo "Creating static library"
ar rcs liball.a *.o
ar -t liball.a 
nm liball.a
mv liball.so liball.bak
gcc test.c -L. -lall -o test-static-lib
#echo "ldd test-static-lib:"
mv liball.bak liball.so
ldd test-static-lib
#echo "ldd test-dynamic-lib:"
ldd test-dynamic-lib
ls -l
