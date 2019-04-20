#!/usr/bin/python3

import os
import sys
import re

fp = None
fp1 = None 
buffer = None

try:
	if not sys.argv[1]:
		print("You need to input file name: python3 ", sys.argv[0], " <filename>")
		sys.exit(1)
	
	if not re.search("\.py$", sys.argv[1]):
		print("Does not end with .py, Please input python file.")	
		sys.exit(1)

	if re.search("3\.py$", sys.argv[1]):
		print("It is likely already a converted file as ending with 3.py. If you are sure, please rename so that it does not end with 3.py")
		sys.exit(1)
	
	fp = open(sys.argv[1])
	if not fp:
		print("Failed to open ", sys.argv[1])
		sys.exit(1)

	outputFile = sys.argv[1][:-3] + "3.py".strip()
	print("output: ", outputFile)
	fp1 = open(outputFile, 'w')
	
	buffer = 1
	
	while buffer:
		buffer = fp.readline()
		#print("read: ", buffer)

		if re.search("print ", buffer):
			line = re.sub("print ", "print(", buffer).strip()
			line = line + ")"
			print(line)	
			fp1.write(line)	
		else:
			fp1.write(buffer)

except Exception as msg:
	print(msg)
 
fp.close()
fp1.close()

os.system("mv " + sys.argv[1] + " " + sys.argv[1][:-3] + "2.py".strip())
os.system("mv " + outputFile + " " + sys.argv[1])

	

