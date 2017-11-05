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
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import make_csv

def prediction_with_random_forest(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:-1], df.iloc[:, -1], test_size = 0.30, random_state = 0)
    random_forest_classifier(X_train, y_train, X_test, y_test)


def prediction_with_tree_classifier(df):
    clf = ExtraTreesClassifier()
    clf = clf.fit(df.iloc[:, 1:-1],df.iloc[:, -1])
    clf.feature_importances_
    model = SelectFromModel(clf, threshold="mean", prefit=True)
    X_new = model.transform(df.iloc[:, 1:-1])
    print df.iloc[:, 1:-1].shape
    print X_new.shape

    X_train, X_test, y_train, y_test = train_test_split(X_new, df.iloc[:, -1], test_size = 0.30, random_state = 0)
    random_forest_classifier(X_train, y_train, X_test, y_test)


def prediction_with_tree_classifier_split(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:-1], df.iloc[:, -1], test_size = 0.30, random_state = 0)
    clf = ExtraTreesClassifier()
    clf = clf.fit(X_train, y_train)
    clf.feature_importances_
    
    model = SelectFromModel(clf, threshold="mean", prefit=True)
    X_train_new = model.transform(X_train)
    print X_train_new.shape
    X_test_new = model.transform(X_test)
    print X_test_new.shape

    random_forest_classifier(X_train_new, y_train, X_test_new, y_test)


def random_forest_classifier(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators= 10, max_features='auto', criterion="gini")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print "F1 score",f1_score(y_test,preds ,average='macro')
    scores=cross_val_score(clf,df.iloc[:, 1:-1], df.iloc[:, -1],cv=5,scoring='f1_macro')
    print "F1 scores with 5 fold cross validation", scores
    print "mean of score i.e accuracy", scores.mean()
    print "accuracy score without k-fold", accuracy_score(preds,y_test)


if __name__=='__main__':
    raw_path_string = raw_input("Enter path where data is located (Location of accession number dirs): ")
    csv_path = raw_input("Enter path of directory to store csv files: ")
    train_path = raw_input("Enter path of train csv file (Path upto p1_train.csv): ")


    make_csv.make_csv_files(raw_path_string, csv_path)
    
    colnames1 = ['TPM']
    slash = "/"
    classifier_input = list()
    
    label_dict = {}
        
    train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')
    for i, row in train_data.iterrows():        
        label_dict[row[0]] = row[1]

    classifier_input = list()

    print "Starting reading csv files"
    files = listdir(csv_path)
    for file in files:
        print csv_path + slash + file
        name = file.split('.')[0]
        data = pd.read_csv(csv_path + slash + file, usecols=colnames1,converters={'TPM': float})
        data_list = [file] + data.TPM.tolist()
        
        classifier_label = label_dict[name]
        data_list = data_list + [classifier_label]
        classifier_input.append(data_list)

print "Read all csv files, creating dataframe"

df = pd.DataFrame(classifier_input)
print "Created dataframe"

# prediction_with_random_forest(df)
# prediction_with_tree_classifier(df)
prediction_with_tree_classifier_split(df)