#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from numpy import log10 as log
from sklearn.utils import shuffle


# In[2]:


class wordClass:
    spamCount = 0
    hamCount = 0
    spamProbability = 0
    hamProbability = 0

    def __init__(self, name):
        self.name = name


# In[3]:


df = pd.read_csv("emails.csv", header=None, names=['message', 'label'])
df = shuffle(df)

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'])

vectorizer = CountVectorizer()
fittedVectorizerXTrain = vectorizer.fit_transform(X_train)
trainingData = fittedVectorizerXTrain.toarray()
trainingDataFeatureNames = vectorizer.get_feature_names_out()

vectorizer2 = CountVectorizer()
fittedVectorizerXTest = vectorizer2.fit_transform(X_test)
testData = fittedVectorizerXTest.toarray()
testDataFeatureNames = vectorizer2.get_feature_names_out()

# In[4]:


trainingWords = {}
for i in range(len(trainingDataFeatureNames)):
    trainingWords[trainingDataFeatureNames[i]] = wordClass(str(trainingDataFeatureNames[i]))

# In[5]:


for i in range(trainingData.shape[0]):
    for y in range(trainingData.shape[1]):
        if trainingData[i][y] > 0:
            if y_train.values[i] == "0":
                trainingWords[trainingDataFeatureNames[y]].hamCount += 1
            else:
                trainingWords[trainingDataFeatureNames[y]].spamCount += 1

# In[6]:


for i in range(len(trainingWords.keys())):
    if trainingWords[trainingDataFeatureNames[i]].hamCount == 0 or trainingWords[trainingDataFeatureNames[i]].spamCount == 0:
        trainingWords[trainingDataFeatureNames[i]].hamCount += 1
        trainingWords[trainingDataFeatureNames[i]].spamCount += 1

# In[7]:


trainingDataSpamCount = 0
trainingDataHamCount = 0
for i in range(len(y_train.values)):
    if y_train.values[i] == "0":
        trainingDataHamCount += 1
    else:
        trainingDataSpamCount += 1

# In[8]:


trainingDataSpamProbability = log(trainingDataSpamCount / trainingData.shape[0])
trainingDataHamProbability = log(trainingDataHamCount / trainingData.shape[0])

# In[9]:


for i in range(len(trainingWords)):
    trainingWords[trainingDataFeatureNames[i]].spamProbability = log(
        trainingWords[trainingDataFeatureNames[i]].spamCount / trainingDataSpamCount)
    trainingWords[trainingDataFeatureNames[i]].hamProbability = log(
        trainingWords[trainingDataFeatureNames[i]].hamCount / trainingDataHamCount)

# In[15]:


TP = 0
TN = 0
FP = 0
FN = 0
for i in range(testData.shape[0]):
    result = "0"
    probabilityOfSpam = 0
    probabilityOfHam = 0
    for y in range(testData.shape[1]):
        if testDataFeatureNames[y] in trainingWords.keys():
            probabilityOfSpam += trainingWords[trainingDataFeatureNames[y]].spamProbability
            probabilityOfHam += trainingWords[trainingDataFeatureNames[y]].hamProbability
            break

    probabilityOfSpam += trainingDataSpamProbability
    probabilityOfHam += trainingDataHamProbability

    if probabilityOfSpam > probabilityOfHam:
        result = "1"

    if y_test.values[i] == result:
        if result == "1":
            TP += 1
        else:
            TN += 1
    else:
        if result == "1":
            FP += 1
        else:
            FN += 1

# In[16]:


TP

# In[17]:


TN

# In[18]:


FP

# In[19]:


FN

# In[ ]:

print("a")
