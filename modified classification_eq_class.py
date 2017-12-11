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
from sklearn import tree
import graphviz
import make_csv
import datetime
import parse_eq_classes

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
    # random_forest_classifier(X_new, y, to_predict)
    decision_tree_classifier(df.iloc[:, 1:-2], y, X_new, to_predict)


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


def decision_tree_classifier(X, y, X_new, to_predict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    #clf = tree.DecisionTreeClassifier(random_state=0, max_features=None)
    clf = tree.DecisionTreeClassifier(max_features = None, criterion = 'entropy', splitter = 'best', max_depth = None, min_samples_split = 2, min_samples_leaf = 1)
    fit_model = clf.fit(X_train, y_train)
    output_pred = fit_model.predict(X_test)
    print("Prediction: ", output_pred)
    print("F1 score predicted w/o cross val", f1_score(y_test, output_pred, average='weighted'))

    dot_data = tree.export_graphviz(fit_model, out_file=None, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("output_1")

    # f1 score
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    print "F1 scores with 5 fold cross validation with reduced features for ", to_predict, scores
    print "F1 score", scores.mean()

    # accuracy
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print "accuracy scores with 5 fold cross validation with reduced features for ", to_predict, scores
    print "mean of accuracy", scores.mean()


if __name__=='__main__':
    # raw_path_string = raw_input("Enter path where data is located (Location of accession number dirs): ")
    # csv_path = raw_input("Enter path of directory to store csv files: ")
    # train_path = raw_input("Enter path of train csv file (Path upto p1_train.csv): ")
    slash = "/"
    raw_path_string = '/home/rasika/Documents/Computational Biology/Project/Data'
    csv_path = '/home/rasika/Documents/Computational Biology/Project/Result'
    train_path = '/home/rasika/Documents/Computational Biology/Project/p1_train_pop_lab.csv'

    # make csv files from quant.sf files
    make_csv.make_csv_files(raw_path_string + slash, csv_path, slash)
    
    colnames1 = ['Name', 'TPM']
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

    accession_numbers = parse_eq_classes.get_features()
    seen = False
    files = listdir(csv_path)
    for file in files:
        name = file.split('.')[0]
        data = pd.read_csv(csv_path + slash + file, usecols=colnames1, converters={'TPM': float})
        
        unique_transcripts = accession_numbers[name]
        transcripts = list(unique_transcripts.keys())
        # print(data.head())
        # print(data.TPM.tolist())
        tpms = []

        # tpms.append(data['TPM'].where(data['Name'] in transcripts))

        # tpms.append(data.loc[data['Name'].isin(transcripts)])
        # print('tpms :: ', len(tpms))

        data = pd.DataFrame([unique_transcripts])
        print(data.head(1))
        data_list = [name] + data[:, ].tolist()
        # data_list = [name] + tpms
        # if not seen:
        #     print('data list', data_list)
        #     seen = True

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
# prediction_with_tree_classifier(df, 1) # 1 for predicting sequence center
# prediction_with_pca(df)
# prediction_with_Kbest(df)
print datetime.datetime.now()
