FILENAME=p125
OUTPUT=$FILENAME.out
/usr/local/cuda-9.2/bin/nvcc -arch=sm_35 -rdc=true $FILENAME.cu -o $OUTPUT -lcudadevrt
./$OUTPUT
#nvvp ./$OUTPUT
