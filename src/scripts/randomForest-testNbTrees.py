'''
Main file
'''

from sklearn.ensemble import RandomForestClassifier
from tools import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    print "---BEGIN---"
    
    trainingSet_fileName = "data/train500.csv"
    nbTreesRange = range(1, 100, 1)
    
    # get the Training and CV sets
    (trainingSet_df, cvSet_df) = getTrainingAndCvSets(trainingSet_fileName, splitLevel=70, randomize=False)
    trainingSet_features = trainingSet_df.drop('label',1)
    trainingSet_label = trainingSet_df['label']
    cvSet_features = cvSet_df.drop('label',1)
    cvSet_label = cvSet_df['label']
    
    trainingSet_results = numpy.array( [] )
    cvSet_results = numpy.array( [] )
    
    for nbTrees in nbTreesRange:
        # Train the Random Forest
        rf = RandomForestClassifier(n_estimators=nbTrees, n_jobs=2, verbose=0)
    
        print "Training Random Forest with",nbTrees,"trees"
        rf.fit(numpy.asarray(trainingSet_features), numpy.asarray(trainingSet_label))
        print "End of Training"
        
        # Predict on the training set
        score = computeScore(rf, trainingSet_features, trainingSet_label, "Training Set")
        trainingSet_results = numpy.append(trainingSet_results, [1 - score])
        
        # Predict on the CV set
        score = computeScore(rf, cvSet_features, cvSet_label, "CV Set")
        cvSet_results = numpy.append(cvSet_results, [1 - score])
        
    
    #plot the results
    plt.plot(nbTreesRange, trainingSet_results)
    plt.plot(nbTreesRange, cvSet_results)
    plt.show()
    
    print "---END---"
    
    
    
    