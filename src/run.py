'''
Main file
'''

from sklearn.ensemble import RandomForestClassifier
from tools import *

if __name__ == '__main__':
    
    print "---BEGIN---"
    
    # get the Training and CV sets
    (trainingSet_df, cvSet_df) = getTrainingAndCvSets('data/train500.csv', splitLevel=70)
    trainingSet_features = trainingSet_df.drop('label',1)
    trainingSet_label = trainingSet_df['label']
    cvSet_features = cvSet_df.drop('label',1)
    cvSet_label = cvSet_df['label']
    
    # Train the Random Forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2, verbose=2)

    print "Training Random Forest..."
    rf.fit(numpy.asarray(trainingSet_features), numpy.asarray(trainingSet_label))
    print "End of Training"
    
    # Predict on the training set
    score = computeScore(rf, trainingSet_features, trainingSet_label, "Training Set")
    
    # Predict on the CV set
    score = computeScore(rf, cvSet_features, cvSet_label, "CV Set")

    # Run on test set
    runOnTestSet(rf, 'data/test.csv', 'data/test_result.csv')
    
    print "---END---"
    
    
    
    