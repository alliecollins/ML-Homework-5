'''
Citation: Prof Ghani's repo
I am using the mlfunctions as a baseline for creating a config file here to feed into my pipeline
for purposes of this assignment (e.g. the models required as per class)
'''


# Import Statements
from __future__ import division
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_params():
    '''
    Note: I have modified to be the classifiers we discussed in class / were noted as 
    required in HW3

    I have also modified the parameters I change to mirror those we discussed/I incorporated
    in last homework (vs. what was in the original file in prof Ghani's repo)
    '''
    clfs = {
    'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'LR': LogisticRegression(penalty='l1', C=1e5),
    'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
    'GB': GradientBoostingClassifier(max_depth=4, n_estimators=10),
    'DT': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'BAG': BaggingClassifier()
            }
    
    grid = { 
    'RF':{'n_estimators': [100, 1000], 'max_depth': [5,50], 'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.01,1,10]},
    'GB': {'n_estimators': [10, 50], 'max_depth': [5,10]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.01,1]},
    'KNN' :{'n_neighbors': [5,50],'metric': ["euclidean", "minkowski"]},
    'BAG':{'n_estimators':[5,10], 'max_samples': [10, 20]}
           }
    
    return clfs, grid
