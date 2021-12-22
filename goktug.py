# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:51:49 2021

@author: gcand
"""
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from sklearn.feature_extraction.text import CountVectorizer

start = time.time()

"""Functions For Categorical Estimation"""


def getProbabilityForCategoricalData(label_train):
    # label_train = y_train for categorical value
    labels = label_train.tolist()
    total = len(labels)

    probabilityOfBooksReview = labels.count("books") / total
    probabilityOfMusicReview = labels.count("music") / total
    probabilityOfHealthReview = labels.count("health") / total
    probabilityOfDvdReview = labels.count("dvd") / total
    probabilityOfCameraReview = labels.count("camera") / total
    probabilityOfSoftwareReview = labels.count("software") / total

    return probabilityOfBooksReview, probabilityOfMusicReview, probabilityOfHealthReview, probabilityOfDvdReview, probabilityOfCameraReview, probabilityOfSoftwareReview, total


def laplaceSmoothing(Bow, sentencestoPredict, totalWordCount):
    numberOfSpecifiedCategoricalWord = len(Bow)
    myList = []
    for word in sentencestoPredict:
        if word in Bow.keys():
            count = Bow[word]
        else:
            count = 0
        myList.append((count + 1) / (totalWordCount + numberOfSpecifiedCategoricalWord))
    resultDict = dict(zip(sentencestoPredict, myList))

    return resultDict


def myNaiveBayesForCategoricalData(sentencestoPredict):
    PoBeingBook, PoBeingMusic, PoBeingHealth, PoBeingDvd, PoBeingCamera, PoBeingSoftware, totalWordCount = getProbabilityForCategoricalData(
        y_train)

    bookDict = laplaceSmoothing(bookBow, sentencestoPredict, totalWordCount)
    musicDict = laplaceSmoothing(musicBow, sentencestoPredict, totalWordCount)
    healthDict = laplaceSmoothing(healthBow, sentencestoPredict, totalWordCount)
    dvdDict = laplaceSmoothing(dvdBow, sentencestoPredict, totalWordCount)
    cameraDict = laplaceSmoothing(cameraBow, sentencestoPredict, totalWordCount)
    softwareDict = laplaceSmoothing(softwareBow, sentencestoPredict, totalWordCount)

    probabilityOfBeingBook = calculateProbability(bookDict, PoBeingBook)
    probabilityOfBeingMusic = calculateProbability(musicDict, PoBeingMusic)
    probabilityOfBeingHealth = calculateProbability(healthDict, PoBeingHealth)
    probabilityOfBeingDvd = calculateProbability(dvdDict, PoBeingDvd)
    probabilityOfBeingCamera = calculateProbability(cameraDict, PoBeingCamera)
    probabilityOfBeingSoftware = calculateProbability(softwareDict, PoBeingSoftware)

    result = max(probabilityOfBeingBook, probabilityOfBeingMusic, probabilityOfBeingHealth, probabilityOfBeingDvd,
                 probabilityOfBeingCamera, probabilityOfBeingSoftware)

    if (result == probabilityOfBeingBook):
        return "book"
    elif (result == probabilityOfBeingMusic):
        return "music"
    elif (result == probabilityOfBeingHealth):
        return "health"
    elif (result == probabilityOfBeingDvd):
        return "dvd"
    elif (result == probabilityOfBeingCamera):
        return "camera"
    else:
        return "software"


"""Functions For Sentiment Estimation """


def getProbabilityOfBeingNegativeOrPositive():
    labels = y_train.tolist()
    numberOfPositiveReview = labels.count("pos")
    numberOfNegativeReview = labels.count("neg")

    total = numberOfPositiveReview + numberOfNegativeReview

    probabilityOfBeingPositive = numberOfPositiveReview / total
    probabilityOfBeingNegative = numberOfNegativeReview / total

    return probabilityOfBeingPositive, probabilityOfBeingNegative, total


def calculateProbability(wordDictionary, probability):
    pb = 0
    for key in wordDictionary.keys():
        pb = pb + math.log(wordDictionary[key])
    pb = pb + math.log(probability)
    return pb


def createBoW(vec, X):
    word_list = vec.get_feature_names()
    count_list = X.toarray().sum(axis=0)
    Bow = dict(zip(word_list, count_list))
    return Bow


def prepareBowForNaiveBayes():
    vec_pos = CountVectorizer()
    X_Positive = vec_pos.fit_transform(positiveReview)

    vec_neg = CountVectorizer()
    X_Negative = vec_neg.fit_transform(negativeReview)

    word_list_positive = vec_pos.get_feature_names()
    count_list_p = X_Positive.toarray().sum(axis=0)
    positive_Bow = dict(zip(word_list_positive, count_list_p))

    word_list_negative = vec_neg.get_feature_names()
    count_list_n = X_Negative.toarray().sum(axis=0)
    negative_Bow = dict(zip(word_list_negative, count_list_n))

    # For categorical estimation
    vec_book = CountVectorizer()
    X_Book = vec_book.fit_transform(booksReview)

    vec_camera = CountVectorizer()
    X_Camera = vec_camera.fit_transform(cameraReview)

    vec_dvd = CountVectorizer()
    X_Dvd = vec_dvd.fit_transform(dvdReview)

    vec_music = CountVectorizer()
    X_Music = vec_music.fit_transform(musicReview)

    vec_health = CountVectorizer()
    X_Health = vec_health.fit_transform(healthReview)

    vec_software = CountVectorizer()
    X_Software = vec_software.fit_transform(softwareReview)

    bookBow = createBoW(vec_book, X_Book)
    cameraBow = createBoW(vec_camera, X_Camera)
    dvdBow = createBoW(vec_dvd, X_Dvd)
    musicBow = createBoW(vec_music, X_Music)
    healthBow = createBoW(vec_health, X_Health)
    softwareBow = createBoW(vec_software, X_Software)

    return positive_Bow, negative_Bow, bookBow, cameraBow, dvdBow, musicBow, healthBow, softwareBow


def myNaiveBayes(sentencestoPredict, positive_Bow, negative_Bow):
    PoBeingPositive, PoBeingNegative, totalWordCount = getProbabilityOfBeingNegativeOrPositive()

    # Laplace Smooting
    positiveDict = laplaceSmoothing(positive_Bow, sentencestoPredict, totalWordCount)
    negativeDict = laplaceSmoothing(negative_Bow, sentencestoPredict, totalWordCount)

    probabilityOfBeingPositive = calculateProbability(positiveDict, PoBeingPositive)
    probabilityOfBeingNegative = calculateProbability(negativeDict, PoBeingNegative)

    if (probabilityOfBeingPositive > probabilityOfBeingNegative):
        return "pos"
    else:
        return "neg"


def get_accuracy(cm):
    total = 0
    nTruePredict = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            total = total + cm[i][j]
            if (i == j):
                nTruePredict = nTruePredict + cm[i][j]
    accuracy = 100 * nTruePredict / total
    return accuracy


data = pd.read_csv("all_sentiment_shuffled.txt", sep="\t",
                   names=['tokens', 'topic', 'sentiment', 'document identifier'])
data[['topic', 'sentiment']] = data['tokens'].str.split(' ', 1, expand=True)
data[['sentiment', 'document identifier']] = data['sentiment'].str.split(' ', 1, expand=True)
data[['document identifier', 'tokens']] = data['document identifier'].str.split(' ', 1, expand=True)

review = []
positiveReview = []
negativeReview = []
booksReview = []
cameraReview = []
dvdReview = []
healthReview = []
musicReview = []
softwareReview = []

from nltk.corpus import stopwords
import re

for i in range(len(data["tokens"])):
    wordList = re.sub('[^a-zA-Z-" "-"\'"]', "", data["tokens"].iloc[i]).split()
    wordList = [word for word in wordList if not word in set(stopwords.words("english"))]

    wordList = " ".join(wordList)
    data["tokens"].iloc[i] = wordList
    if (data["sentiment"].iloc[i] == "neg"):
        negativeReview.append(wordList)
    else:
        positiveReview.append(wordList)

    review.append(wordList)

    # Categorical part
    if (data["topic"].iloc[i] == "books"):
        booksReview.append(wordList)
    elif (data["topic"].iloc[i] == "camera"):
        cameraReview.append(wordList)
    elif (data["topic"].iloc[i] == "dvd"):
        dvdReview.append(wordList)
    elif (data["topic"].iloc[i] == "health"):
        healthReview.append(wordList)
    elif (data["topic"].iloc[i] == "music"):
        musicReview.append(wordList)
    else:
        softwareReview.append(wordList)

end = time.time()
print("Dosya Okumak icin Gecen zaman", (end - start) / 60, " dakika")
start = time.time()

# Shuffle the data. Creating test and train data
from sklearn.utils import shuffle

data = shuffle(data)

data_size = len(data)
test_size = int(data_size / 5)
train_size = (data_size - test_size)

train_data = data[:train_size]
x_train = data[:train_size]
x_test = data[train_size:]
y_train = data.sentiment[:train_size]
y_test = data.sentiment[train_size:]
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,data["sentiment"],test_size=0.20,random_state=23)
"""
de = []
y_pred = []

bookBow = {}
cameraBow = {}
dvdBow = {}
healthBow = {}
musicBow = {}
softwareBow = {}

positiveBow, negativeBow, bookBow, cameraBow, dvdBow, musicBow, healthBow, softwareBow = prepareBowForNaiveBayes()
for i in range(len(y_test)):
    # for i in range(1):
    de = (x_test["tokens"].iloc[i]).split()
    sonucc = myNaiveBayes(de, positiveBow, negativeBow)
    y_pred.append(sonucc)

# accuracy
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy for sentiment : ", get_accuracy(cm))

from collections import Counter

possitiveComment = dict(Counter(positiveBow).most_common(20))
negativeComment = dict(Counter(negativeBow).most_common(20))

data = shuffle(data)

data_size = len(data)
test_size = int(data_size / 5)
train_size = (data_size - test_size)

train_data = data[:train_size]
x_train = data[:train_size]
x_test = data[train_size:]
y_train = data.topic[:train_size]
y_test = data.topic[train_size:]

# x_train,x_test,y_train,y_test = train_test_split(data,data["topic"],test_size=0.20,random_state=23)
y_pred2 = []
for i in range(len(y_test)):
    de = (x_test["tokens"].iloc[i]).split()
    sonucc = myNaiveBayesForCategoricalData(de)
    y_pred2.append(sonucc)

cm = confusion_matrix(y_test, y_pred2)
cm = np.array(cm)

cm = np.delete(cm, 0, 0)
cm = np.delete(cm, 1, 1)
print(cm)
print("Accuracy for categorical estimation : ", get_accuracy(cm))

end = time.time()
print("Gecen zaman", (end - start) / 60, " dakika")