import re
import os
import time
#import docx

fp=""

#docx.Document(filename)

#for p in docx.paragraphs:
#	print(p)
		

# fail because antiword not available.		
#import textract
#text = textract.process(filename)

#import docx2txt
#readblob=docx2txt.process(filename)
#print(readblob)

csv_data={}
	
counter=0
entry_data_candidate=[]
csv_data={}
counter_entry=0
file_counter=0
flag_valid_entry=0

csv_data['date']=[]

for filename in os.listdir():
	if re.search("\.txt", filename):
		print("Found ", filename)
		
		fp=open(filename)
		if not fp:
			print("Can not open ", filename)
			quit(1)
		
		csv_data['date'].append(filename.split('.')[0])
		content=fp.read().split('\n')

		for i in content:
			# 	Build entries
			#	Requirement.
			#	Read until ???? line, if encountered, start counter.
			#	if read, start counter until next, assume 4 lines. (1. entry 2. cost 3. discount 4. cost)
			#	number of lines counter
			#	if counter = 4.
			#		assign to dictionary.
			#	dictionary struct:
			#	construct csv
			#	csv format:
			#	--- 		date1, 	date2...
			#	entry1,		total,	total...
			# 	python format
			#	list\
			#	{'date':  [date1,date2...], \
			#	'entry1': [total_date1, total_date2]\
			#	'entry2': [total_date1, total_date2]..]
			#	--
			#	entry1
			#	entry2
			#...
			#	entryN
			
			print(i)
			time.sleep(0)
			if re.search("\?\?\?", i):
				print("Checking if passed through valid entry... counter:", counter)
				
				if counter == 4 or counter == 6:
					print("Reached new entry, perform additional check.")
					#	Assume valid entry
					flag_valid_entry=1
					print("candidate:", entry_data_candidate)
					#print("candidate:" , entry_data_candidate[1], entry_data_candidate[2], entry_data_candidate[3])
					for i in range(1,4):
						#if not type(re.sub(",", "", entry_data_candidate[i].strip()))==int and not type(re.sub(",", "", entry_data_candidate[i].strip()))==float:
						#	print("Disqualified for valid entry: ", entry_data_candidate[i].strip())
						#	counter = 0
						#	flag_valid_entry=0
						try:
							float(re.sub(",","", entry_data_candidate[i].strip()))
						except Exception as msg:
							print("Disqualified for valid entry: ", entry_data_candidate[i].strip())
							counter = 0
							flag_valid_entry=0
							
					if flag_valid_entry:	
						print("Qualified for valid entry")
					else:
						print("Disqualified for valid entry, skipping...")
						entry_data_candidate=[]
						continue
					
					#	if date row is added which 1st row and nothing else.
					#	then we append new row as entry.
					
					if not entry_data_candidate[0] in csv_data:
						print(entry_data_candidate[0], ", does not exist yet, adding...")
						csv_data[entry_data_candidate[0]]=[]
						
						#	This will take care of situation where newer service on the day other than first date.
						for i in range(0, file_counter):
								csv_data[entry_data_candidate[0]].append(0)
					
					try:
						csv_data[entry_data_candidate[0]].append(float(re.sub(",","",(entry_data_candidate[3].strip()))))
					except Exception as msg:
						print("FATAL Error, ENCOUNTERED VALID ENTRY CANDIDATE BUT TOTAL FIELD IS NOT FLOAT!!! CHECK YOUR DATA OR CODE!")
						quit(1)
									
					counter_entry+=1
				counter = 0
				entry_data_candidate=[]
			else:
				entry_data_candidate.append(i)
				counter += 1
				
		#	if subsequent date did not perform service, then detect by checking all dictionary list sizes are
		#	filecounter size. If not add one 0 item signaling, there is not service for that item.
		
		for key in csv_data.keys():
			if len(csv_data[key])-1 == file_counter:
				print(key, ": OK")
			elif len(csv_data[key])-1 == file_counter-1:
				print("Adding 0 amount")
				csv_data[key].append(0)
			else:
				print("FATAL Error: the length of ", key, " list ", len(csv_data[key])-1,", must be either equal to current file counter: ", file_counter, " or one less. Logic error! Can not continue...")
				#quit(1)
		file_counter+=1		
		
print("------------------------------------------")		
for i in csv_data:
	print(i, csv_data[i])
print("------------------------------------------")		
			
			
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		