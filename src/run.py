'''
Main file
'''

from pandas import *
import random

trainingSet_Ratio = 70 #ratio in % of the training set to use for training

if __name__ == '__main__':
    
    #Read the CSV files which contains training examples
    df = read_csv('data/train500.csv')
    print "CSV file size: ",df.shape[0],"x",df.shape[1]
    
    # Compute the size of trainingSet and cvSet
    trainingSet_size = int(round(df.shape[0] * trainingSet_Ratio /100))
    cvSet_size = df.shape[0] - trainingSet_size
    print "trainingSet size:",trainingSet_size
    print "cvSet size:",cvSet_size
    
    
    #split the dataframe into the training set and the cv set
    rowsIndexes = random.sample(df.index, trainingSet_size)
    trainingSet_df = df.ix[rowsIndexes]
    cvSet_df = df.drop(rowsIndexes)
    
    print trainingSet_df.shape
    print cvSet_df.shape
    
    
    