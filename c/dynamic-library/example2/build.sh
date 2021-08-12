# dynamic build...

gcc demo.c -o demo_dyn
readelf -h demo_dyn
ldd demo_dyn

# static build...
gcc -static demo.c -o demo_stc
readelf -h demo_stc
ldd demo_stc

# using lib overload.

gcc preload.c -o preload.so -fPIC -shared -ldl

echo "Preloading with demo_dyn..."
LD_PRELOAD="./preload.so" ./demo_dyn
echo "Preloading with demo_stc..."
LD_PRELOAD="./preload.so" ./demo_stc
