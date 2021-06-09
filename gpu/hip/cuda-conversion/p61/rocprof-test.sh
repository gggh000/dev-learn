if [[ -z `which csvtool` ]] ; then apt install csvtool -y ; fi
FILE=p61-rocprof
hipcc $FILE.cpp
OUTPUT=rocprof-test.csv

if [[ $? -eq 0 ]] ; then
    rocprof -i input.xml -o $OUTPUT ./a.out
    csvtool readable $OUTPUT
else 
    echo "Compile failed."
fi
