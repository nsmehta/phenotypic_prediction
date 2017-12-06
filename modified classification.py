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
from sklearn.feature_selection import SelectFromModel, f_classif, chi2
from sklearn.feature_selection import SelectKBest
import make_csv
import datetime


# method for predicting only with random forest classifier
def prediction_with_random_forest(df):
    random_forest_classifier(df.iloc[:, 1:-1], df.iloc[:, -1])


# method for predicting with pca
def prediction_with_pca(df):
    nf = 1000
    pca = PCA(n_components=nf, svd_solver='full', random_state=0)
    # pca.fit(df.iloc[:, 1:-1])

    X_new = pca.fit_transform(df.iloc[:, 1:-1])
    print X_new.shape
    random_forest_classifier(X_new, df.iloc[:, -1])


# method for selecting features with k-best
def prediction_with_Kbest(df):
    test = SelectKBest(chi2, k=3000)
    fit = test.fit(df.iloc[:, 1:-1], df.iloc[:, -1])
    features = fit.transform(df.iloc[:, 1:-1])

    # perform classification using the selected features
    random_forest_classifier(features, df.iloc[:, -1])


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


if __name__=='__main__':
    raw_path_string = raw_input("Enter path where data is located (Location of accession number dirs): ")
    csv_path = raw_input("Enter path of directory to store csv files: ")
    train_path = raw_input("Enter path of train csv file (Path upto p1_train.csv): ")
    slash = "/"

    # make csv files from quant.sf files
    make_csv.make_csv_files(raw_path_string + slash, csv_path, slash)
    
    colnames1 = ['TPM']
    classifier_input = list()

    label_dict = {}
    # store the labels from train file in a dictionary
    train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')
    for i, row in train_data.iterrows():        
        label_dict[row[0]] = (row[1], row[2])

    classifier_input = list()

    print "Starting reading csv files"
    print datetime.datetime.now()

    # create dataframe from all csv files
    # each row corresponds to one accession number and the columns are TPM values of each transcript

    files = listdir(csv_path)
    for file in files:
        name = file.split('.')[0]
        data = pd.read_csv(csv_path + slash + file, usecols=colnames1, converters={'TPM': float})
        data_list = [file] + data.TPM.tolist()

        classifier_population = label_dict[name][0]
	classifier_sequence_center = label_dict[name][1]
        data_list = data_list + [classifier_population, classifier_sequence_center]
        classifier_input.append(data_list)

print "Read all csv files, creating dataframe"
print datetime.datetime.now()

# created dataframe will have following format

# Name       TPM_1  TPM_2  TPM_3  TPM_4 ....  TPM_199324  label
# ERR188021  value  value  value  value ....     value     TSI
# ERR188022    .      .      .      .   ....       .       CEU
#   .          .      .      .      .   ....       .        .
#   .          .      .      .      .   ....       .        .
#   .          .      .      .      .   ....       .        .

df = pd.DataFrame(classifier_input)


print "Created dataframe"
print datetime.datetime.now()

# execute classifier
prediction_with_tree_classifier(df, 0) # 0 for predicting population
prediction_with_tree_classifier(df, 1) # 1 for predicting sequence center
# prediction_with_pca(df)
# prediction_with_Kbest(df)
print datetime.datetime.now()
