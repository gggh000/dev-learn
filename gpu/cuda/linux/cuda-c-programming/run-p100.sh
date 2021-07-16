FILENAME=p100   
BAR="----------------------------------"
OUTPUT=$FILENAME.out
LOGFILE=./p100.log
function printWithBar() {
echo $BAR
echo $1
echo $BAR 
}

x=( 32 32 16 16 )
y=( 32 16 32 16 )

nvcc -g -G -arch=sm_20 $FILENAME.cu -o $OUTPUT
if [[ $? -ne 0 ]] ; then echo "Compilation failed!" ; exit 1 ; fi
size=$((${#x[@]}-1))
echo -ne "" > $LOGFILE

echo "running with ${x[$i]} : ${y[$i]}" | tee -a $LOGFILE
printWithBar "running with no profiler" | tee -a $LOGFILE
for i in $(seq 0 $size)
do
    ./$OUTPUT ${x[$i]} ${y[$i]}  | tee -a $LOGFILE
done

echo "running with ${x[$i]} : ${y[$i]}" | tee -a $LOGFILE
printWithBar "achived_occupancy" | tee -a $LOGFILE
for i in $(seq 0 $size)
do
    nvprof --metrics achieved_occupancy ./$OUTPUT ${x[$i]} ${y[$i]}  | tee -a $LOGFILE
done

echo "running with ${x[$i]} : ${y[$i]}" | tee -a $LOGFILE
printWithBar "--events branch,divergent_branch" | tee -a $LOGFILE
for i in $(seq 0 $size)
do
    nvprof --events branch,divergent_branch  ./$OUTPUT ${x[$i]} ${y[$i]}  | tee -a $LOGFILE
done

echo "running with ${x[$i]} : ${y[$i]}"  | tee -a $LOGFILE
printWithBar "--metrics gld_throughput"  | tee -a $LOGFILE
for i in $(seq 0 $size)
do
    nvprof --metrics gld_throughput ./$OUTPUT ${x[$i]} ${y[$i]} | tee -a $LOGFILE
done

echo "running with ${x[$i]} : ${y[$i]}"  | tee -a $LOGFILE
printWithBar "--metrics branch_efficiency"  | tee -a $LOGFILE
for i in $(seq 0 $size)
do
    nvprof --metrics branch_efficiency ./$OUTPUT ${x[$i]} ${y[$i]} | tee -a $LOGFILE
done


