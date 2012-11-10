from pandas import *
import random
#from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy

def getTrainingAndCvSets(fileName, splitLevel=100, randomize=True):
    
    # Read the CSV file which contains training examples
    print "Reading CSV file..."
    df = read_csv(fileName)
    print "CSV file size: ",df.shape[0],"x",df.shape[1]
    
    # Compute the size of trainingSet and cvSet
    trainingSet_size = int(round(df.shape[0] * splitLevel /100))
    cvSet_size = df.shape[0] - trainingSet_size
    print "Training Set size:",trainingSet_size
    print "CV Set size:",cvSet_size
    
    # Split the dataframe into the training set and the cv set
    if randomize == True:
        rowsIndexes = random.sample(df.index, trainingSet_size)
    else:
        rowsIndexes = range(0, trainingSet_size)
    trainingSet_df = df.ix[rowsIndexes]
    cvSet_df = df.drop(rowsIndexes)
    
    return (trainingSet_df, cvSet_df)

def computeScore(classifier, Xset, Yset, setName=""):
    predVal = classifier.predict(numpy.asarray(Xset))
    score = metrics.precision_score(Yset, predVal)
    print "Score",setName,":",score
    
    return score
    
def runOnTestSet(classifier, fileNameTestSet, fileNameOutput):
    # Read the test set
    print "Reading Test set..."
    testSet_df = read_csv(fileNameTestSet)
    print "CSV file size: ",testSet_df.shape[0],"x",testSet_df.shape[1]
    
    # Compute the size of test set
    testSet_size = testSet_df.shape[0]
    print "Test Set size:",testSet_size
    
    # Predict on the test set
    print "Predicting Test Set"
    testSet_predVal = classifier.predict(numpy.asarray(testSet_df))
    
    # Write output file for test set
    print "Writing Test set results"
    numpy.savetxt(fileNameOutput,testSet_predVal, fmt="%1d")