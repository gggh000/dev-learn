./clean.sh
FILE1=p61
FILE2=p61-2kernels
FILE3=p61-2streams
FILE4=p61-2streams-event
for FILE in $FILE4 ; do
    echo "Processing for $FILE.cpp"
    hipcc $FILE.cpp -o $FILE.out
    #rocprof --hip-trace --hsa-trace --roctx-trace -i input.xml -d ./prof ./$FILE.out 
    rocprof --hip-trace -d ./$FILE ./$FILE.out
done
