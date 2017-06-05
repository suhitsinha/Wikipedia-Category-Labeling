
# coding: utf-8

# In[ ]:

from model2 import Model2
from DataParser import DataParser


# In[ ]:

maxParagraphLength=150
maxParagraphs=7
labels=10000
vocabularySize=99636
model = Model2(maxParagraphLength,maxParagraphs,labels, vocabularySize)
training = DataParser(maxParagraphLength,maxParagraphs,labels)
training.getDataFromfile("data/vocab_3L_l10000_red_train.txt")

testing = DataParser(maxParagraphLength,maxParagraphs,labels)
testing.getDataFromfile("data/vocab_3L_l10000_red_test.txt")

epoch=50
for e in range(epoch):
    print 'Epoch: ' + str(e)
    for itr in range(training.totalPages):
        cost=model.train(training.nextBatch())
    print str(cost)
    model.save("model2_l10000_"+str(e))

'''
model.load("results/model2_9")

# In[ ]:

for itr in range(testing.totalPages):
    bp, pp = model.predict(testing.nextBatch())
    print str(bp.shape)
    print str(pp.shape)
'''
