import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import graphviz
import pydot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
from sklearn.dummy import DummyClassifier

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

def create_binary_col(df, col, criteria_dict):
	'''
	Maps values of a df column to 1 or 0 as outlined in the criteria_dict, taken
	as an input to the function
	'''
	df[col+'_binary'] = df[col].map(criteria_dict)
	return df

def discretize_categorical(df, cols):
	'''
	Takes a categorical variable and turns it into dummy columns to use in models. Cols must be a list,
	even if just one column
	'''
	df = pd.get_dummies(df, columns=cols)
	return df

###### BUILD CLASSIFIER ######

def temporal_split(df, start_col, end_col, start_date, end_date, dep_var, predictors=None):
	'''
	Takes in a date range for the testing data; training data is all data prior to 
	the start date

	Testing data is that which is bounded between the start and end date
	'''
	start_date = pd.to_datetime(start_date)
	end_date = pd.to_datetime(end_date)

	train = df[df[end_col] < start_date]
	test = df[(df[start_col] >= start_date) & (df[end_col] <= end_date)]

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

	return (x, x_train, x_test, y_train, y_test)

def create_decision_tree(df, dep_var, max_depth, min_samples_split, file_name, predictors=None, threshold=.5, 
	temporal=False, start_col=None, end_col=None, start_date=None, end_date=None):
	'''
	Create a decision tree using sklearn. Requires pandas dataframe, list of
	predictors to use as input. If no predictors are input, it defaults to using
	all potential predictors

	Creates separate training, testing data either using sklearn default or making
	a temporal split as above

	Returns predicted y-values and y-testing values; also creates dot file to
	visualize tree
	'''

	#Generate the testing and training data
	if temporal:
		x, x_train, x_test, y_train, y_test = temporal_split(df, start_col, end_col, start_date, 
			end_date, dep_var, predictors)

	else:
		y = df[dep_var]

		if not predictors:
			x = df.drop(dep_var, axis=1)
		
		else:
			x = df[predictors]

		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
	
	#Create the model
	model = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
	model = model.fit(x_train, y_train)
	y_scores = model.predict_proba(x_test)
	y_predict = [1 if x[1]>threshold else 0 for x in y_scores]

	tree.export_graphviz(model, out_file=file_name, feature_names=x.columns)

	return (y_test, y_predict, y_scores, model.feature_importances_)

def knn_classify(df, dep_var, n_neighbors, metric, predictors=None, threshold=.5, temporal=False, start_col=None, 
	end_col=None, start_date=None, end_date=None):
	'''
	Create a nearest neighbor model using sklearn. Requires pandas dataframe, number of neighbors,
	list of predictors, dependent variable to use as input. If no predictors are input, it defaults 
	to using all potential predictors.

	Creates separate training, testing data either using sklearn default or making
	a temporal split as above.

	Returns predicted y-values and y-testing values.
	'''

	if temporal:
		x, x_train, x_test, y_train, y_test = temporal_split(df, start_col, end_col, start_date, 
			end_date, dep_var, predictors)

	else:
		y = df[dep_var]

		if not predictors:
			x = df.drop(dep_var, axis=1)
		
		else:
			x = df[predictors]

	knn = KNeighborsClassifier(n_neighbors, metric= metric)
	knn.fit(x_train, y_train)
	y_scores = knn.predict_proba(x_test)
	y_predict = [1 if x[1]>threshold else 0 for x in y_scores]
	params = knn.get_params()

	return (y_test, y_predict, y_scores, params)

def logistic_classify(df, dep_var, penalty, c_value, predictors=None, threshold=.5, temporal=False, 
	start_col=None, end_col=None, start_date=None, end_date=None):
	'''
	Create a logistic regression model using sklearn. Requires pandas dataframe, penalty type, 
	c (tradeoff between low training and testing errors,list of predictors, dependent variable 
	to use as input. If no predictors are input, it defaults to using all potential predictors.

	Creates separate training, testing data either using sklearn default or making
	a temporal split as above.

	Returns predicted y-values and y-testing values.
	'''

	if temporal:
		x, x_train, x_test, y_train, y_test = temporal_split(df, start_col, end_col, start_date, 
			end_date, dep_var, predictors)

	else:
		y = df[dep_var]

		if not predictors:
			x = df.drop(dep_var, axis=1)
		
		else:
			x = df[predictors]

	lr = LogisticRegression(random_state=0, solver='liblinear', penalty=penalty, C=c_value)
	lr.fit(x_train, y_train)
	y_scores = lr.predict_proba(x_test)
	y_predict = [1 if x[1]>threshold else 0 for x in y_scores]
	coeffs = lr.coef_

	return (y_test, y_predict, y_scores, coeffs)

def SVM_classify(df, dep_var, c_value, predictors=None, threshold=.5, temporal=False, start_col=None, 
	end_col=None, start_date=None, end_date=None):
	'''
	Create a support vector model using sklearn. Requires pandas dataframe, c (tradeoff between low
	training and testing errors,list of predictors, dependent variable to use as input. If no predictors 
	are input, it defaults to using all potential predictors.

	Creates separate training, testing data either using sklearn default or making
	a temporal split as above.

	Returns predicted y-values and y-testing values.
	'''

	if temporal:
		x, x_train, x_test, y_train, y_test = temporal_split(df, start_col, end_col, start_date, 
			end_date, dep_var, predictors)

	else:
		y = df[dep_var]

		if not predictors:
			x = df.drop(dep_var, axis=1)
		
		else:
			x = df[predictors]

	svm = LinearSVC(random_state=0, tol=1e-5, C=c_value)
	svm.fit(x_train, y_train)
	y_scores = svm.decision_function(x_test)
	y_predict = [1 if x > threshold else 0 for x in y_scores]

	return (y_test, y_predict, y_scores)

def random_forest(df, dep_var, max_depth=None, predictors=None, threshold=.5, temporal=False, start_col=None, 
	end_col=None, start_date=None, end_date=None):
	'''
	Create a random forest model using sklearn. Requires pandas dataframe, max_depth,list of predictors, 
	dependent variable to use as input. If no predictors are input, it defaults to using all potential predictors.

	Creates separate training, testing data either using sklearn default or making
	a temporal split as above.

	Returns predicted y-values and y-testing values.
	'''

	if temporal:
		x, x_train, x_test, y_train, y_test = temporal_split(df, start_col, end_col, start_date, 
			end_date, dep_var, predictors)

	else:
		y = df[dep_var]

		if not predictors:
			x = df.drop(dep_var, axis=1)
		
		else:
			x = df[predictors]

	clf = RandomForestClassifier(max_depth=max_depth, bootstrap=True)
	clf.fit(x_train, y_train)
	y_scores = clf.predict_proba(x_test)
	y_predict = [1 if x[1]>threshold else 0 for x in y_scores]
	features = clf.feature_importances_

	return (y_test, y_predict, y_scores, features)

def gradient_boost(df, dep_var, max_features=None, predictors=None, threshold=.5, temporal=False, start_col=None, 
	end_col=None, start_date=None, end_date=None):
	'''
	Create a gradient boosting model using sklearn. Requires pandas dataframe, max features,list of predictors, 
	dependent variable to use as input. If no predictors are input, it defaults to using all potential predictors.

	Creates separate training, testing data either using sklearn default or making
	a temporal split as above.

	Returns predicted y-values and y-testing values.
	'''

	if temporal:
		x, x_train, x_test, y_train, y_test = temporal_split(df, start_col, end_col, start_date, 
			end_date, dep_var, predictors)

	else:
		y = df[dep_var]

		if not predictors:
			x = df.drop(dep_var, axis=1)
		
		else:
			x = df[predictors]

	gbc = GradientBoostingClassifier(max_features=max_features)
	gbc.fit(x_train, y_train)
	y_scores = gbc.predict_proba(x_test)
	y_predict = [1 if x[1]>threshold else 0 for x in y_scores]

	return (y_test, y_predict, y_scores)

def evaluate_model(y_test, y_predict):
	'''
	Takes output of create decision tree to evaluate its accuracy and precision
	'''
	accuracy = accuracy_score(y_test, y_predict)
	precision = precision_score(y_test, y_predict)
	fpr, tpr, thresholds = roc_curve(y_test, y_predict)
	model_auc = auc(fpr, tpr)
	recall = recall_score(y_test, y_predict)
	f1 = f1_score(y_test, y_predict)

	return (accuracy, precision, model_auc, recall, f1)

def dummy_baseline(df, dep_var, predictors=None, threshold=.5, temporal=False, start_col=None, 
	end_col=None, start_date=None, end_date=None):
	'''
	Use sklearn's baseline function
	'''
	if temporal:
		x, x_train, x_test, y_train, y_test = temporal_split(df, start_col, end_col, start_date, 
			end_date, dep_var, predictors)

	else:
		y = df[dep_var]

		if not predictors:
			x = df.drop(dep_var, axis=1)
		
		else:
			x = df[predictors]
	
	dummy = DummyClassifier()
	dummy.fit(x_train, y_train)
	y_scores = dummy.predict_proba(x_test)
	y_predict = [1 if x[1]>threshold else 0 for x in y_scores]

	return (y_test, y_predict, y_scores)

def precision_recall(y_test, y_score):
	'''
	Creates precision recall curve; code provided in scikit learn function
	'''
	precision, recall, _ = precision_recall_curve(y_test, y_score)
	average_precision = average_precision_score(y_test, y_score)

	step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
	plt.figure(figsize=(20,10))
	plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
