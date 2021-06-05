if [[ -z `which csvtool` ]] ; then apt install csvtool -y ; fi
hipcc p61.cpp
OUTPUT=rocprof-test.csv
rocprof -i input.txt -o $OUTPUT ./a.out
csvtool readable $OUTPUT
