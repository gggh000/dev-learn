nvcc -g -G -arch=sm_20 p84.cu -o p84.out
#echo "BRANCH OCCUPANCY:"
#ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./p84.out
#echo "BRANCH EFFICIENCY:"
#ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./p84.out

nvprof --events branch,divergent_branch  ./a.out
nvprof --metrics gld_throughput ./a.out
nvprof --metrics branch_efficiency ./a.out
