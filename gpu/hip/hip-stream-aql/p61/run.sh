./clean.sh
FILE1=p61
FILE2=p61-2kernels
FILE3=p61-2streams
for FILE in $FILE3 ; do
    echo "Processing for $FILE.cpp"
    hipcc $FILE.cpp -o $FILE.out
    #rocprof --hip-trace --hsa-trace --roctx-trace -i input.xml -d ./prof ./$FILE.out 
    rocprof --hip-trace --hsa-trace -d ./$FILE ./$FILE.out
done
