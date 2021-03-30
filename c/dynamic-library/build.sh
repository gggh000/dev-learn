CONFIG_PATH_USE_RPATH=1
CONFIG_PATH_USE_LD_LIBRARY_PATH=2
CONFIG_PATH_USE_LDCONFIG=3
CONFIG_PATH_OPTION=$CONFIG_PATH_USE_LDCONFIG

echo Creating  *.o files...
gcc lib*.c -c -fPIC

echo Generating *.so library...
gcc *.o -shared -o liball.so

echo Linking and test...

if [[ $CONFIG_PATH_OPTION -eq $CONFIG_PATH_USE_LD_LIBRARY_PATH ]] ;  then
    echo "Using rpath..."
    gcc -L`pwd`-rpath=`pwd`test.c -lall -o test
    -rpath=/home/username/foo
elif [[ $CONFIG_PATH_OPTION -eq $CONFIG_PATH_USE_LDCONFIG ]] ; then
    echo "Using LDCONFIG to set library path."
    cp liball.so /usr/lib
    chmod 755 /usr/lib/liball.so
    ldconfig
    ldconfig -p | grep liball
    gcc test.c -lall -o test
    ldd test | grep all
else
    echo "Using LD_LIBRARY_PATH to set library path."
    gcc -L`pwd` test.c -lall -o test
    export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
fi
