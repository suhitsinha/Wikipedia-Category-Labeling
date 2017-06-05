import numpy as np
import json
import sys

def CompressData(infile, outfile):
    f=open(infile)
    word_voc = set()
    label_voc = set()
    totalPages = int(f.readline())
    for p in range(totalPages):
        pageId= f.readline()
        if not pageId :
            break
        labelCount=int(f.readline())
        for i in range(labelCount):
            tempLab=int(f.readline())
            label_voc.add(tempLab)
        instancesCount=int(f.readline())
        for i in range(instancesCount):
            tempInstance=f.readline().split()
            temp=[int(x) for x in tempInstance]
            for t in temp:
                word_voc.add(t)
    f.close()
        
    id_mapping = {old_id: new_id for (new_id, old_id) in enumerate(word_voc)}
    label_mapping = {old_label: new_label for (new_label, old_label) in enumerate(label_voc)}

    json.dump(id_mapping, open(outfile+'_id_mapping.json', 'w'))
    json.dump(label_mapping, open(outfile+'_label_mapping.json', 'w'))

    with open(infile) as fin:
        with open(outfile+'.txt', 'w') as fout:
            totalPages = int(fin.readline())
            fout.write(str(totalPages)+'\n')
            for p in range(totalPages):
                pageId = fin.readline().strip()
                if not pageId:
                    break
                fout.write(pageId.strip()+'\n')
                labelCount = int(fin.readline())
                fout.write(str(labelCount)+'\n')    
                for i in range(labelCount):
                    l = int(fin.readline())
                    fout.write(str(label_mapping[l])+'\n')
                instancesCount = int(fin.readline())
                fout.write(str(instancesCount)+'\n')
                for i in range(instancesCount):
                    tempInstance=fin.readline().split()
                    tempInstance=[int(x) for x in tempInstance]
                    for t in tempInstance:
                        fout.write(str(id_mapping[t])+' ')
                    fout.write('\n')

    return len(id_mapping), len(label_mapping)

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    w_count, label_count = CompressData(infile, outfile)
    print 'Word count: ' + str(w_count) + ' Label count: ' + str(label_count)
