
# coding: utf-8

# In[ ]:

from model import Model
from DataParser import DataParser


# In[ ]:

maxParagraphLength=100
maxParagraphs=7
labels=5000
model = Model(maxParagraphLength,maxParagraphs,labels)
training = DataParser(maxParagraphLength,maxParagraphs,labels)
training.getDataFromfile("data/trainSmall.txt")

testing = DataParser(maxParagraphLength,maxParagraphs,labels)
testing.getDataFromfile("data/testSmall.txt")


# In[ ]:
epoch=10
for e in range(epoch):
    print 'Epoch: ' + str(e)
    for itr in range(training.totalPages):
        model.train(training.nextBatch())
    model.save("results/model_"+str(e))

'''
model.load("results/experiment1")

# In[ ]:

for itr in range(testing.totalPages):
    print model.predict(testing.nextBatch())
'''
