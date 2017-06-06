
# coding: utf-8

# In[ ]:

from model2 import Model2
from DataParser import DataParser


# In[ ]:

maxParagraphLength=250
maxParagraphs=10
labels=1000
vocabularySize=150000
model = Model2(maxParagraphLength,maxParagraphs,labels,vocabularySize)
training = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
training.getDataFromfile("data/vocab_3L_l1000_red_train.txt")

testing = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
testing.getDataFromfile("data/vocab_3L_l1000_red_test.txt")

epoch=5
for e in range(epoch):
    print 'Epoch: ' + str(e)
    for itr in range(training.totalPages):
        cost=model.train(training.nextBatch())
    print str(cost)
    model.save("model2_l1000_"+str(e))

'''
model.load("results/model2_9")

# In[ ]:

for itr in range(testing.totalPages):
    bp, pp = model.predict(testing.nextBatch())
    print str(bp.shape)
    print str(pp.shape)
'''
