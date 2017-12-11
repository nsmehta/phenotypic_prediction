import re
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from sklearn.feature_extraction import DictVectorizer
import os
from os import listdir
from scipy.cluster.hierarchy import ward, dendrogram,linkage,fcluster,cophenet,distance
import scipy.cluster.hierarchy as hier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, f_classif, chi2
from sklearn.feature_selection import SelectKBest
from sklearn import tree
import make_csv
import datetime


# method for predicting only with random forest classifier
def prediction_with_random_forest(df):
    random_forest_classifier(df.iloc[:, 1:-1], df.iloc[:, -1])


# method for selecting features with extra trees classifier
def prediction_with_tree_classifier(df, option):
    clf = ExtraTreesClassifier(random_state=0)
    to_predict = None
    if option == 0: # to predict population
	   y = df.iloc[:, -2]
	   to_predict = 'Population'
    else: # to predict sequence center
	   y = df.iloc[:, -1]
	   to_predict = 'Sequence center'

    clf = clf.fit(df.iloc[:, 1:-2], y)
    model = SelectFromModel(clf, threshold="mean", prefit=True)
    X_new = model.transform(df.iloc[:, 1:-2])

    # perform classification using the selected features
    random_forest_classifier(X_new, y, to_predict)
    decision_tree_classifier_population(df.iloc[:, 1:199325], y, X_new, to_predict)
    decision_tree_classifier_center(df.iloc[:, 199325:-2], y, X_new, to_predict)

    # decision_tree_classifier_multi(df.iloc[:, 1:-2], df.iloc[:, -2:], X_new, to_predict)

# classify the data to predict labels using random forest classifier
def random_forest_classifier(X_train, y_test, to_predict):
    # changed the max_features to None
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators= 10, max_features=None, criterion="gini")

    # f1 score
    scores = cross_val_score(clf, X_train, y_test, cv=5, scoring='f1_macro')
    print "F1 scores with 5 fold cross validation for ", to_predict, scores
    print "F1 score", scores.mean()

    # accuracy
    scores = cross_val_score(clf, X_train, y_test, cv=5, scoring='accuracy')
    print "accuracy scores with 5 fold cross validation for ", to_predict, scores
    print "mean of accuracy", scores.mean()


def decision_tree_classifier_population(X, y, X_new, to_predict):
    print "population",X.head()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    #clf = tree.DecisionTreeClassifier(random_state=0, max_features=None)
    clf = tree.DecisionTreeClassifier(max_features = None, criterion = 'entropy', splitter = 'best', max_depth = None, min_samples_split = 2, min_samples_leaf = 1)
    fit_model = clf.fit(X_train, y_train)
    output_pred = fit_model.predict(X_test)
    print("Prediction: ", output_pred)
    print("F1 score predicted w/o cross val", f1_score(y_test, output_pred, average='weighted'))

    # f1 score
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    print "F1 scores with 5 fold cross validation for ", to_predict, scores
    print "F1 score", scores.mean()

    # accuracy
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print "accuracy scores with 5 fold cross validation for ", to_predict, scores
    print "mean of accuracy", scores.mean()



def decision_tree_classifier_center(X, y, X_new, to_predict):
    print "Sequence center",X.head()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    #clf = tree.DecisionTreeClassifier(random_state=0, max_features=None)
    clf = tree.DecisionTreeClassifier(max_features = None, criterion = 'entropy', splitter = 'best', max_depth = None, min_samples_split = 2, min_samples_leaf = 1)
    fit_model = clf.fit(X_train, y_train)
    output_pred = fit_model.predict(X_test)
    print("Prediction: ", output_pred)
    print("F1 score predicted w/o cross val", f1_score(y_test, output_pred, average='weighted'))

    # f1 score
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    print "F1 scores with 5 fold cross validation for ", to_predict, scores
    print "F1 score", scores.mean()

    # accuracy
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print "accuracy scores with 5 fold cross validation for ", to_predict, scores
    print "mean of accuracy", scores.mean()


if __name__=='__main__':
    raw_path_string = raw_input("Enter path where data is located (Location of accession number dirs): ")
    csv_path = raw_input("Enter path of directory to store csv files: ")
    train_path = raw_input("Enter path of train csv file (Path upto p1_train.csv): ")
    slash = "\\"
    # raw_path_string = '/home/rasika/Documents/Computational Biology/Project/Data'
    # csv_path = '/home/rasika/Documents/Computational Biology/Project/Result'
    # train_path = '/home/rasika/Documents/Computational Biology/Project/p1_train_pop_lab.csv'


    # make csv files from quant.sf files
    colnames1 = ['TPM', 'Length']
    make_csv.make_csv_files(raw_path_string + slash, csv_path, slash,['Name']+colnames1)


    classifier_input = list()

    label_dict = {}
    # store the labels from train file in a dictionary
    train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')
    for i, row in train_data.iterrows():
        label_dict[row[0]] = (row[1], row[2])

    classifier_input = list()

    print "Starting reading csv files"
    print datetime.datetime.now()


    files = listdir(csv_path)
    for file in files:
        name = file.split('.')[0]
        data = pd.read_csv(csv_path + slash + file, usecols=colnames1, converters={'TPM': float,'Length':float})
        data_list = [file] + data.TPM.tolist()+data.Length.tolist()

        classifier_population = label_dict[name][0]
	classifier_sequence_center = label_dict[name][1]
        data_list = data_list + [classifier_population, classifier_sequence_center]
        classifier_input.append(data_list)

print "Read all csv files, creating dataframe"
print datetime.datetime.now()

df = pd.DataFrame(classifier_input)


print "Created dataframe"
print datetime.datetime.now()

#prediction_with_tree_classifier(df, 0) # 0 for predicting population
prediction_with_tree_classifier(df, 1) # 1 for predicting sequence center
print datetime.datetime.now()
