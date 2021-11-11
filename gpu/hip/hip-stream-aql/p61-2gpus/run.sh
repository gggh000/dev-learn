./clean.sh
hipcc p61.cpp
#rocprof --hip-trace --hsa-trace --roctx-trace -i input.xml -d ./prof ./a.out 
rocprof --hip-trace  -i input.xml -d ./prof ./a.out 
