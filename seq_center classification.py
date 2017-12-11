import re
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from sklearn.feature_extraction import DictVectorizer
import os
from os import listdir
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster, cophenet, distance
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from modified_classification_eq_class import eq_classes

def prediction_with_pca(df):
    print df.shape
    sc = StandardScaler()
    X_std = sc.fit_transform(df.iloc[:, 1:-1])
    pca = PCA(n_components=1000, svd_solver='auto', random_state=0)
    # pca.fit(df.iloc[:, 1:-1])
    y = df.iloc[:, -1]
    y_numerical = pd.factorize(y)[0]
    X_new = pca.fit_transform(X_std)
    random_forest_classifier(X_new, df.iloc[:, -1])
    decision_tree_classifier_seqcenter(X_new, y_numerical)

def crossval(df):
    df = np.array_split(df, 5)
    f1_score1 = []
    accuracy = []

    for i in range(len(df)):
        test = pd.DataFrame(df[i])
        train = df[:i] + df[i + 1:]
        train = pd.concat(train)
        X_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1]
        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]
        clf = tree.DecisionTreeClassifier(random_state=0, max_features=None, criterion='gini', splitter='best',
                                          max_depth=None, min_samples_split=2, min_samples_leaf=5,class_weight='balanced')
        fit_model = clf.fit(X_train, y_train)
        output_pred = fit_model.predict(X_test)
        f1_score1.append(f1_score(y_test, output_pred, average='weighted'))
        accuracy.append(accuracy_score(y_test, output_pred))

    print "F1 scores with 5 fold cross validation for Population",f1_score1
    print "accuracy scores with 5 fold cross validation for Population", accuracy
    f1 = np.mean(f1_score1)
    accuracy1 = np.mean(accuracy)
    print "F1 Score", f1
    print "accuracy", accuracy1

# method for predicting only with random forest classifier
def prediction_with_random_forest(df):
    random_forest_classifier(df.iloc[:, 1:-1], df.iloc[:, -1])


# method for selecting features with extra trees classifier
def prediction_with_tree_classifier(df):
    #Creating a clf and creating a data frame y of only labels and fitting the model using dataframe leavinng first
    # and last column and passing the label df
    clf = ExtraTreesClassifier(random_state=0)
    y = df.iloc[:, -1]
    clf = clf.fit(df.iloc[:, 1:-1], y)
    model = SelectFromModel(clf, threshold="mean", prefit=True)
    X_new = model.transform(df.iloc[:, 1:-1])

    # perform classification using the selected features
    random_forest_classifier(X_new, y)
    decision_tree_classifier_seqcenter(df.iloc[:, 1:-1], y)
    # decision_tree_classifier_multi(df.iloc[:, 1:-2], df.iloc[:, -2:], X_new, to_predict)


# classify the data to predict labels using random forest classifier
def random_forest_classifier(X_train, y_test):
    # changed the max_features to None
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=10, max_features=None, criterion="gini")

    # f1 score
    scores = cross_val_score(clf, X_train, y_test, cv=5, scoring='f1_macro')
    print "F1 scores with 5 fold cross validation for Seq Center RT ", scores
    print "F1 score", scores.mean()

    # accuracy
    scores = cross_val_score(clf, X_train, y_test, cv=5, scoring='accuracy')
    print "accuracy scores with 5 fold cross validation for Seq Center RT", scores
    print "mean of accuracy", scores.mean()


def decision_tree_classifier_seqcenter(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # clf = tree.DecisionTreeClassifier(random_state=0, max_features=None)
    clf = tree.DecisionTreeClassifier(max_features=None, criterion='gini', splitter='best', max_depth=None,
                                      min_samples_split=2, min_samples_leaf=5,class_weight='balanced')
    fit_model = clf.fit(X_train, y_train)
    output_pred = fit_model.predict(X_test)
    # print("Prediction: ", output_pred)
    print("F1 score predicted w/o cross val DT", f1_score(y_test, output_pred, average='weighted'))
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    print "F1 scores with 5 fold cross validation for Population DT", scores
    print "F1 score", scores.mean()

    # accuracy
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print "accuracy scores with 5 fold cross validation for Population DT", scores
    print "mean of accuracy", scores.mean()

    b = np.array([y])
    X = np.hstack((X,b.T))
    df_new = pd.DataFrame(X)

    crossval(df_new)

if __name__ == '__main__':
    # raw_path_string = raw_input("Enter path where data is located (Location of accession number dirs): ")
    # csv_path = raw_input("Enter path of directory to store csv files: ")
    # train_path = raw_input("Enter path of train csv file (Path upto p1_train.csv): ")
    slash = "\\"
    raw_path_string = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\train'
    csv_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\ResultsCenter'
    train_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\p1_train_pop_lab.csv'

#     # make csv files from quant.sf files
#     colnames1 = ['TPM','Length']
#     make_csv.make_csv_files(raw_path_string + slash, csv_path, slash, ['Name'] + colnames1)
#
#     classifier_input = list()
#
#     label_dict = {}
#     # store the labels from train file in a dictionary
#     train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')
#     for i, row in train_data.iterrows():
#         label_dict[row[0]] = (row[1], row[2])
#
#     classifier_input = list()
#
#     print "Starting reading csv files"
#     print datetime.datetime.now()
#     # Reading the data from csv files and creating a data list of acession number tpm and length
#     files = listdir(csv_path)
#     for file in files:
#         name = file.split('.')[0]
#         data = pd.read_csv(csv_path + slash + file, usecols=colnames1, converters={'TPM':float,'Length': float})
#         data_list = [name] + data.TPM.tolist()+data.Length.tolist()
#     # also adding label sequence center to the data list
#         classifier_center = label_dict[name][1]
#         data_list = data_list + [classifier_center]
#         classifier_input.append(data_list)
#
# print "Read all csv files, creating dataframe"
# print datetime.datetime.now()
#
# df = pd.DataFrame(classifier_input)
#
# print "Created dataframe"
# print datetime.datetime.now()
df=eq_classes(raw_path_string, csv_path, train_path, slash)
prediction_with_tree_classifier(df)
print datetime.datetime.now()
