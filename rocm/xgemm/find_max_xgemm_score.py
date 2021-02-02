#!/usr/bin/python3
import sys
import os

try:
	fileName=sys.argv[1]
except Exception as msg:
	print(msg)
	quit(1)

fp=open(fileName)
if not fp:
	print("Error, can not open file: ", fileName)
	quit(1)

lines=fp.readlines()

for i in lines:
	print(i)

scores=[]
max_score=0
max_score_idx=0
max_score_line=0
counter=0

for i in lines:
	if counter != 0:
		scores.append(float(i.split(',')[4]))
		if scores[-1] > max_score:
			max_score = scores[-1]
			max_score_line=i
	counter+=1

max_score_idx=lines.index(max_score_line)

print("Max score (index): ", max_score, "(", max_score_idx, ")")
print(max_score_line)

