import numpy as np
import json
import sys

def CompressData(infile, outfile, train_size):
    count = 0
    f=open(infile)
    fout = open(outfile+'_train.txt', 'w')
    totalPages = int(f.readline())
    assert totalPages > train_size
    fout.write(str(train_size)+'\n')
    for p in range(totalPages):
        if (p == train_size):
            fout.close()
            fout = open(outfile+'_test.txt', 'w')
            fout.write(str(totalPages-train_size)+'\n')
        pageId= f.readline().strip()
        if not pageId :
            break
        fout.write(pageId+'\n')
        labelCount=int(f.readline())
        fout.write(str(labelCount)+'\n')
        for i in range(labelCount):
            tempLab=int(f.readline())
            fout.write(str(tempLab)+'\n')
        instancesCount=int(f.readline())
        fout.write(str(instancesCount)+'\n')
        for i in range(instancesCount):
            tempInstance=f.readline().strip()
            fout.write(tempInstance+'\n')
    f.close()
    fout.close()
        
if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    ts = int(sys.argv[3])
    CompressData(infile, outfile, ts)
