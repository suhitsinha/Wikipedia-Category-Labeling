import numpy as np
import json
import sys
import pickle

def CompressData(infile, outfile, no_labels):
    f=open(infile)
    label_voc = {}
    label_list = []
    totalPages = int(f.readline())
    for p in range(totalPages):
        pageId= f.readline()
        if not pageId :
            break
        labelCount=int(f.readline())
        label_set = set()
        for i in range(labelCount):
            tempLab=int(f.readline())
            label_set.add(tempLab)
            try:
                label_voc[tempLab] += 1
            except KeyError:
                label_voc[tempLab] = 1
        label_list.append(label_set)
        instancesCount=int(f.readline())
        for i in range(instancesCount):
            tempInstance=f.readline()
    f.close()

    sorted_labels = []
    sorted_freq = []
    for key, value in sorted(label_voc.iteritems(), key=lambda(k,v): (v,k)):
        sorted_labels.append(key)
        sorted_freq.append(value)
    selected_labels = set(sorted_labels[:no_labels])
    pickle.dump(sorted_freq, open('freq.pkl', 'w'))

    count = 0
    for i in range(len(label_list)):
        label_list[i].intersection_update(selected_labels)
        if len(label_list[i]) > 0:
            count += 1

    with open(infile) as fin:
        with open(outfile+'_l'+str(no_labels)+'.txt', 'w') as fout:
            totalPages = int(fin.readline())
            fout.write(str(count)+'\n')
            for p in range(totalPages):
                pageId = fin.readline().strip()
                if not pageId:
                    break

                # If label set not null, write all labels
                no_l = len(label_list[p])
                if no_l > 0:
                    fout.write(pageId.strip()+'\n')
                    fout.write(str(no_l)+'\n')
                    for l in label_list[p]:
                        fout.write(str(l)+'\n')

                # Read all labels from the input files
                labelCount = int(fin.readline())
                for i in range(labelCount):
                    l = int(fin.readline())

                instancesCount = int(fin.readline())
                instance_list = []
                for i in range(instancesCount):
                    instance=fin.readline().strip()
                    instance_list.append(instance)

                if no_l > 0:
                    fout.write(str(instancesCount)+'\n')
                    for i in range(instancesCount):
                        fout.write(instance_list[i]+'\n')

    return count

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    no_labels = int(sys.argv[3])
    a_count = CompressData(infile, outfile, no_labels)
    print 'No of articles: ' + str(a_count)
