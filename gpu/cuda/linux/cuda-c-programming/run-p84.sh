nvcc p84.cu -o p84.out
echo "BRANCH OCCUPANCY:"
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./p84.out
echo "BRANCH EFFICIENCY:"
ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./p84.out

