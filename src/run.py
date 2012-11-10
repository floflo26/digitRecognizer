'''
Main file
'''

from pandas import *
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy

trainingSet_Ratio = 70 #ratio in % of the training set to use for training

if __name__ == '__main__':
    
    print "---BEGIN---"
    
    # Read the CSV files which contains training examples
    print "Reading CSV file..."
    df = read_csv('data/train500.csv')
    print "CSV file size: ",df.shape[0],"x",df.shape[1]
    
    # Compute the size of trainingSet and cvSet
    trainingSet_size = int(round(df.shape[0] * trainingSet_Ratio /100))
    cvSet_size = df.shape[0] - trainingSet_size
    print "Training Set size:",trainingSet_size
    print "CV Set size:",cvSet_size
    
    # Split the dataframe into the training set and the cv set
    rowsIndexes = random.sample(df.index, trainingSet_size)
    trainingSet_df = df.ix[rowsIndexes]
    cvSet_df = df.drop(rowsIndexes)
    
    # Train the Random Forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2, verbose=2)
    trainingSet_features = trainingSet_df.drop('label',1)
    trainingSet_label = trainingSet_df['label']
    print "Training Random Forest..."
    rf.fit(numpy.asarray(trainingSet_features), numpy.asarray(trainingSet_label))
    print "End of Training"
    
    # Predict on the training set
    trainingSet_predVal = rf.predict(numpy.asarray(trainingSet_features))
    trainingSet_score = metrics.precision_score(trainingSet_label, trainingSet_predVal)
    print "Score Training Set:",trainingSet_score
    
    # Predict on the CV set
    cvSet_features = cvSet_df.drop('label',1)
    cvSet_label = cvSet_df['label']
    cvSet_predVal = rf.predict(numpy.asarray(cvSet_features))
    cvSet_score = metrics.precision_score(cvSet_label, cvSet_predVal)
    print "Score CV Set:",cvSet_score
    
    # Read the test set
    print "Reading Test set..."
    testSet_df = read_csv('data/test.csv')
    print "CSV file size: ",testSet_df.shape[0],"x",testSet_df.shape[1]
    
    # Compute the size of test set
    testSet_size = testSet_df.shape[0]
    print "Test Set size:",testSet_size
    
    # Predict on the test set
    print "Predicting Test Set"
    testSet_predVal = rf.predict(numpy.asarray(testSet_df))
    
    # Write output file for test set
    print "Writing Test set results"
    numpy.savetxt("data/test_result.csv",testSet_predVal, fmt="%1d")
    
    print "---END---"
    
    
    
    