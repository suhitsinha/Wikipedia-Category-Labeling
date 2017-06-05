
# coding: utf-8

# In[ ]:

from model2 import Model2
from DataParser import DataParser


# In[ ]:

maxParagraphLength=100
maxParagraphs=7
labels=14252
vocabularySize=70597
model = Model2(maxParagraphLength,maxParagraphs,labels, vocabularySize)
training = DataParser(maxParagraphLength,maxParagraphs,labels)
training.getDataFromfile("data/trainSmallRed.txt")

testing = DataParser(maxParagraphLength,maxParagraphs,labels)
testing.getDataFromfile("data/trainSmallRed.txt")

epoch=10
for e in range(epoch):
    print 'Epoch: ' + str(e)
    for itr in range(training.totalPages):
        cost=model.train(training.nextBatch())
    print str(cost)
    model.save("results/model2_"+str(e))

'''
model.load("results/model2_9")

# In[ ]:

for itr in range(testing.totalPages):
    bp, pp = model.predict(testing.nextBatch())
    print str(bp.shape)
    print str(pp.shape)
'''
