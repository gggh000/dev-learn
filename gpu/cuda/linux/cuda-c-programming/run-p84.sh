FILENAME=p84
BAR="----------------------------------"
OUTPUT=$FILENAME.out
nvcc -g -G -arch=sm_20 $FILENAME.cu -o $OUTPUT
#echo "BRANCH OCCUPANCY:"
#ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./$OUTPUT
#echo "BRANCH EFFICIENCY:"
#ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./$OUTPUT

function printWithBar() {
echo $BAR
echo $1
echo $BAR 
}

printWithBar "--events branch,divergent_branch"
nvprof --events branch,divergent_branch  ./$OUTPUT $1 $2
printWithBar "--metrics gld_throughput"
nvprof --metrics gld_throughput ./$OUTPUT $1 $2
printWithBar "--metrics branch_efficiency"
nvprof --metrics branch_efficiency ./$OUTPUT $1 $2

