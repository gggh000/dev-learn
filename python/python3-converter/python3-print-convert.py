import os
import sys

if not sys.argv[1]:
	print("You need to input file name: python3 ", sys.argv[0], " <filename>")
	return 0;

fp = open(sys.argv[1])

if not fp:
	print("Failed to open ", sys.argv[1])
	return 0;

buffer = 1

while buffer:
	buffer = print file.readline()
	print("read: ", buffer)


 
	

