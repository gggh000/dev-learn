fp = open("file.all.log")
fpOut = open("race.found.log", 'w')

if not fp:
    print "Failed to open."

line = fp.readline().strip()
    
counter = 0

while line:
    print line
    operands= line.split(":")[1:4]
    print "operands: ", operands

    if int(operands[0]) + int(operands[1]) != int(operands[2]):
        fpOut.write("RACE!: " + str(operands) + "\n")
        counter += 1

    line = fp.readline().strip()

if counter == 0:
    fpOut.write("No race condition found.")
    print "No race condition found."
else:
    print "At least one race condition found: ", counter

fp.close()
fpOut.close()
