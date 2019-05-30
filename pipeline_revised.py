from __future__ import division 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn import tree
import graphviz
import pydot
from sklearn.utils.fixes import signature
from sklearn.metrics import roc_auc_score
import pipeline_config
import pylab as pl
from datetime import timedelta
import random
from scipy import optimize
import time
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
 
    
###### STEP: READ IN DATA ######

def read_file(filename):
    '''
    Reads in csv file and converts to pandas dataframe
    '''
    df = pd.read_csv(filename)
    return df

###### STEP: DISPLAY SUMMARY STATS #######

def calc_summary_stats(df, cols_to_drop=None, cols_to_include=None):
    '''
    Calculates mean, median, standard deviation, max and min values for all columns 
    (note: this may be non-sensical for categorical variables, so optional parameters
    allow for specification of either which columns should remain or which should
    be eliminated)
    '''
    
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    if cols_to_include:
        df = df[cols_to_include]

    summary_df = pd.DataFrame(df.mean())
    summary_df = summary_df.rename(columns={0:'mean'})
    summary_df['std_dev'] = df.std()
    summary_df['median'] = df.median()
    summary_df['max_val'] = df.max()
    summary_df['min_val'] = df.min()

    return summary_df

def generate_summary_plot(df, dep_var, predictor):
    '''
    Generates simple scatterplot for dependent variable and a given predictor
        to look at correlation / spot any outliers
    '''
    df.plot.scatter(x=predictor, y=dep_var)

###### STEP: PRE-PROCESS DATA ######

def fill_missing(df):
    '''
    fill in missing values with the median value of column
    '''
    df.fillna(df.median(), inplace=True)
    return df

###### GENERATE FEATURES/PREDICTORS ######

def create_binned_col(df, col, bins, include_lowest=True):
    '''
    Takes a continuous variable and creates bins. The labels are simply ints,
    as this is what sklearn decision tree requires.
    '''
    labels = list(range(len(bins)-1))
    df[col+'_binned'] = pd.cut(df[col], bins, labels=labels, include_lowest=include_lowest)
    return df

def create_binary_col(df, map_dict):
    '''
    Maps values of a df column to 1 or 0 as outlined in the criteria in the map_dict of cols to criteria, taken
    as an input to the function
    '''
    cols_to_drop = []

    for mapping in map_dict:
        col, criteria_dict = mapping
        df[col+'_binary'] = df[col].map(criteria_dict)
        cols_to_drop.append(col)

    df = df.drop(cols_to_drop, axis=1)
    return df

def discretize_categorical(df, cols):
    '''
    Takes a categorical variable and turns it into dummy columns to use in models. Cols must be a list,
    even if just one column
    '''
    df = pd.get_dummies(df, columns=cols)
    return df

def scale_data(train_df, test_df, col):
    '''
    Scale data on training and apply to testing set leveraging scikitlearn function
    Inputs: training df, testing df, colum
    '''
    mm_scaler = preprocessing.StandardScaler()
    mm_scaler.fit(train_df_col)

    train_df[col] = mm_scaler.transform(train_df[[col]])
    test_df[col] = mm_scaler.transform(test_df[[col]])

    return train_df, test_df


def check_col_match(train_df, test_df):
    '''
    Since we build dummies and perform other cleaning on train and test separately, this function
    removes any dummy variables created in the test data that do not exist and add in dummy columns not present
    in the testing column that were created in the training data
    '''
    extra_cols = []

    for column in test_df.columns:
        if column not in train_df.columns:
            extra_cols.append(column)

    test_df = test_df.drop(extra_cols,axis=1)

    for column in train_df.columns:
        if column not in test_df.columns:
            test_df[column] = 0

    return train_df, test_df


###### BUILD CLASSIFIERS ######

def temporal_split(df, date_col, test_start, test_set_length, label_duration, dep_var, predictors=None):
    '''
    Takes in a start date for testing data, a size of the test set, and the duration
    being used in the label creation to ensure adequate time  training data is all data prior to 
    the start date

    Testing data is that which is bounded between the start and end date
    '''
    test_start = pd.to_datetime(test_start)
    test_end = test_start + pd.DateOffset(days=test_set_length)
    train_end = test_start + pd.DateOffset(days=label_duration)

    train = df[df[date_col] < train_end]
    test = df[(df[date_col] >= test_start) & (df[date_col] <= train_end)]

    y_train = train[dep_var]
    y_test = test[dep_var]

    if not predictors:
        x = df.drop(dep_var, axis=1)
        x_train = train.drop(dep_var, axis=1)
        x_test = test.drop(dep_var, axis=1)

    else:
        x = df[predictors]
        x_train = train[predictors]
        x_test = test[predictors]

    return (x_train, x_test, y_train, y_test)

def run_models(df_tuple_list, models_to_run='all'):
    '''
    Citation: Prof Ghani's magic loops function, adapted to accomodate my temporal

    Leverages loops to iterate over all parameters; parameters to test are defined in the
    helper function file
    '''
    if models_to_run == 'all':
        models_to_run=['RF','LR','DT','BAG','KNN','GB']

    clfs, grid = pipeline_config.get_params()

    results_df =  pd.DataFrame(columns=('model_type','clf', 'date', 'parameters',
        'train_set_size', 'validation_set_size',
        'baseline','precision_at_5','precision_at_10','precision_at_20','precision_at_30','precision_at_40',
        'precision_at_50','recall_at_5','recall_at_10','recall_at_20','recall_at_30','recall_at_40',
        'recall_at_50','auc-roc'))

    model_num = 0

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            for date_df in df_tuple_list:
                date_name, x_train, x_test, y_train, y_test = date_df
                clf.set_params(**p)
                model_num +=1
                model = clf.fit(x_train, y_train)
                pred_probs = clf.predict_proba(x_test)[::,1] 
                results_df.loc[len(results_df)] = [models_to_run[index],clf, date_name, p,
                len(x_train),len(y_test), 
                precision_at_k(y_test,pred_probs, 100),
                precision_at_k(y_test,pred_probs, 5),
                precision_at_k(y_test,pred_probs, 10),
                precision_at_k(y_test,pred_probs, 20),
                precision_at_k(y_test,pred_probs, 30),
                precision_at_k(y_test,pred_probs, 40),
                precision_at_k(y_test,pred_probs, 50),
                recall_at_k(y_test,pred_probs, 5),
                recall_at_k(y_test,pred_probs, 10),
                recall_at_k(y_test,pred_probs, 20),
                recall_at_k(y_test,pred_probs, 30),
                recall_at_k(y_test,pred_probs, 40),
                recall_at_k(y_test,pred_probs, 50),
                roc_auc_score(y_test, pred_probs)]

                plot_precision_recall_n(
                    y_test, pred_probs, 'graphs/'+models_to_run[index]+'_'+str(model_num), 'save')

                if models_to_run[index] == 'DT':
                    tree.export_graphviz(model, out_file='trees/'+models_to_run[index]+'_'+str(model_num), feature_names=x_train.columns)
                    feature_scores = model.feature_importances_
                    d = {'Features': x_train.columns, "Importance": feature_scores}
                    feature_importance = pd.DataFrame(data=d)
                    feature_importance = feature_importance.sort_values(by=['Importance'], ascending=False)
                    feature_importance.to_csv('features/'+models_to_run[index]+'_'+str(model_num)+'.csv')        
        model_num = 0

    results_df.to_csv('model_results.csv')

    return None
# def run_models(x_train, x_test, y_train, y_test,models_to_run='all'):
#     '''
#     Citation: Prof Ghani's magic loops function

#     Leverages loops to iterate over all parameters; parameters to test are defined in the
#     helper function file
#     '''
#     if models_to_run == 'all':
#         #models_to_run=['RF','LR','DT','SVM','GB','BAG','KNN']
#         models_to_run=['DT','LR','BAG']

#     clfs, grid = pipeline_config.get_params()

#     results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters',
#         'train_set_size', 'validation_set_size',
#         'baseline','precision_at_5','precision_at_10','precision_at_20','precision_at_30','precision_at_40',
#         'precision_at_50','recall_at_5','recall_at_10','recall_at_20','recall_at_30','recall_at_40',
#         'recall_at_50','auc-roc'))

#     model_num = 0

#     for index,clf in enumerate([clfs[x] for x in models_to_run]):
#         parameter_values = grid[models_to_run[index]]
#         for p in ParameterGrid(parameter_values):
#                 clf.set_params(**p)
#                 model_num +=1
#                 model = clf.fit(x_train, y_train)
#                 pred_probs = clf.predict_proba(x_test)[::,1] 
#                 results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
#                 len(x_train),len(y_test), 
#                 precision_at_k(y_test,pred_probs, 100),
#                 precision_at_k(y_test,pred_probs, 5),
#                 precision_at_k(y_test,pred_probs, 10),
#                 precision_at_k(y_test,pred_probs, 20),
#                 precision_at_k(y_test,pred_probs, 30),
#                 precision_at_k(y_test,pred_probs, 40),
#                 precision_at_k(y_test,pred_probs, 50),
#                 recall_at_k(y_test,pred_probs, 5),
#                 recall_at_k(y_test,pred_probs, 10),
#                 recall_at_k(y_test,pred_probs, 20),
#                 recall_at_k(y_test,pred_probs, 30),
#                 recall_at_k(y_test,pred_probs, 40),
#                 recall_at_k(y_test,pred_probs, 50),
#                 roc_auc_score(y_test, pred_probs)]

#                 plot_precision_recall_n(
#                     y_test, pred_probs, 'graphs/'+models_to_run[index]+'_'+str(model_num), 'save')

#                 if models_to_run[index] == 'DT':
#                     tree.export_graphviz(model, out_file='trees/'+models_to_run[index]+'_'+str(model_num), feature_names=x_train.columns)
#                     feature_scores = model.feature_importances_
#                     d = {'Features': x_train.columns, "Importance": feature_scores}
#                     feature_importance = pd.DataFrame(data=d)
#                     feature_importance = feature_importance.sort_values(by=['Importance'], ascending=False)
#                     feature_importance.to_csv('features/'+models_to_run[index]+'_'+str(model_num)+'.csv')        
#         model_num = 0

#     results_df.to_csv('model_results.csv')

#     return None

def plot_roc(name, probs, true, output_type):
    '''
    CITATION: Prof Ghani's magic loop functions
    Inputs:
    --name: figure name
    --probs: predicted scores
    --true: true test data scores
    --output_type = show or save the chart
    '''
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def generate_binary_at_k(y_scores, k):
    '''
    CITATION: Prof Ghani's magic loop functions
    Take the probability scores and a cutoff for top k to determine the 1 or 0 label
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary

def precision_at_k(y_true, y_scores, k):
    '''
    CITATION: Prof Ghani's magic loop functions
    calculates precision at given threshold
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    '''
    CITATION: Prof Ghani's magic loop functions
    Calculates recall at given threshold
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    '''
    CITATION: Prof Ghani's magic loop functions
    Plots the precision and recall at different thresholds limits
    '''
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def joint_sort_descending(l1, l2):
    '''
    CITATION: Prof Ghani's magic loop functions
    Helper function for preceding pipeline functions
    '''
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]
