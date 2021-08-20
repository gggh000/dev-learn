import re
import time

def process_params(pSysArg, pList):
    DEBUG=1
    ret=[]
    for i in pSysArg:
        if DEBUG:
            print("Processing ", i)
        try:
            for j in pList:
                if re.search(j, i + "="):
                    try:
                        ret.append(int(i.split('=')[1]))
                    except Exception as msg:
                        ret.append(i.split('=')[1])
                
        except Exception as msg:
            print("Error processing ", i)
            print(msg)

    if DEBUG:   
        print("returning ", ret)

    return ret

        
