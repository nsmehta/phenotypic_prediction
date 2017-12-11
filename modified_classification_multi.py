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
from sklearn.preprocessing import StandardScaler
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
# import graphviz
import make_csv
import datetime
from sklearn.decomposition import PCA
from modified_classification_eq_class import eq_classes


def prediction_with_pca(df):
    print df.shape
    sc = StandardScaler()
    X_std = sc.fit_transform(df.iloc[:, 1:-2])
    pca = PCA(n_components=0.9, svd_solver='full', random_state=0)
    # pca.fit(df.iloc[:, 1:-1])
    y = df.iloc[:, -2:]
    # y_numerical = pd.factorize(y)[0]
    X_new = pca.fit_transform(X_std)
    # random_forest_classifier(X_new, df.iloc[:, -1])
    decision_tree_classifier_multi(X_new, y)


def crossval(df):
    stacked = df.iloc[:, -2:].stack()
    df.iloc[:, -2:] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    df = np.array_split(df, 5)
    f1_score1_p = []
    accuracy_p = []
    f1_score1_c=[]
    accuracy_c=[]
    for i in range(len(df)):
        test = pd.DataFrame(df[i])
        train = df[:i] + df[i + 1:]
        train = pd.concat(train)
        X_train = train.iloc[:, 1:-2]
        y_train = train.iloc[:, -2:]
        X_test = test.iloc[:, 1:-2]
        y_test =test.iloc[:, -2:]
        clf = tree.DecisionTreeClassifier(random_state=0, max_features=None, criterion='gini', splitter='best',
                                          max_depth=None, min_samples_split=2, min_samples_leaf=5, class_weight='balanced')
        fit_model = clf.fit(X_train, y_train)
        output_pred= fit_model.predict(X_test)
        f1_score1_p.append(f1_score(y_test.iloc[:,0], output_pred[:,0], average='weighted'))
        accuracy_p.append(accuracy_score(y_test.iloc[:,0], output_pred[:,0]))
        f1_score1_c.append(f1_score(y_test.iloc[:,1], output_pred[:, 1], average='weighted'))
        accuracy_c.append(accuracy_score(y_test.iloc[:,1], output_pred[:,1]))

    print "F1 scores with 5 fold cross validation for Population",f1_score1_p
    print "accuracy scores with 5 fold cross validation for Population", accuracy_p
    f1 = np.mean(f1_score1_p)
    accuracy1 = np.mean(accuracy_p)
    print "F1 Score p", f1
    print "accuracy P", accuracy1

    print "F1 scores with 5 fold cross validation for Seq C", f1_score1_c
    print "accuracy scores with 5 fold cross validation for Seq C", accuracy_c
    f1_c = np.mean(f1_score1_c)
    accuracy1_c = np.mean(accuracy_c)
    print "F1 Score c", f1_c
    print "accuracy c", accuracy1_c

# method for selecting features with extra trees classifier
def prediction_with_tree_classifier(df):
    # perform classification using the selected features
    decision_tree_classifier_multi(df.iloc[:, 1:-2], df.iloc[:, -2:])


def decision_tree_classifier_multi(X, y_2d):
    y_numerical = y_2d.apply(lambda x: pd.factorize(x)[0])
    print y_numerical.head()
    print X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y_numerical, test_size=0.33, random_state=42)

    # clf = tree.DecisionTreeClassifier(random_state=0, max_features=None)
    clf = tree.DecisionTreeClassifier(random_state=0, max_features=None, criterion='gini', splitter='best',
                                      max_depth=None, min_samples_split=2, min_samples_leaf=5, class_weight='balanced')
    fit_model = clf.fit(X_train, y_train)
    output_pred = fit_model.predict(X_test)
    print("Prediction: ", output_pred)
    print("F1 score predicted for population w/o cross val",
          f1_score(y_test.iloc[:, 0], output_pred[:, 0], average='weighted'))
    print("F1 score predicted for sequence center w/o cross val",
          f1_score(y_test.iloc[:, 1], output_pred[:, 1], average='weighted'))

    print(X.shape)
    print(y_numerical.shape)
    # b = np.array([y_numerical])
    X = np.hstack((X, y_numerical))
    df_new = pd.DataFrame(X)

    crossval(df_new)

    # crossval(df)
    # vals = y_2d.stack().drop_duplicates().values
    # b = [x for x in y_2d.stack().drop_duplicates().rank(method='dense')]
    # d1 = dict(zip(b, vals))
    # print (d1)

    # dot_data = tree.export_graphviz(fit_model, out_file=None, filled=True, rounded=True, special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("output_1")

    # f1 score
    # scores = cross_val_score(clf, X, y_numerical, cv=5, scoring='f1_macro')
    # print "F1 scores with 5 fold cross validation with reduced features ", scores
    # print "F1 score", scores.mean()
    #
    # # accuracy
    # scores = cross_val_score(clf, X, y_numerical, cv=5, scoring='accuracy')
    # print "accuracy scores with 5 fold cross validation with reduced features ", scores
    # print "mean of accuracy", scores.mean()


if __name__ == '__main__':
    raw_path_string = raw_input("Enter path where data is located (Location of accession number dirs): ")
    csv_path = raw_input("Enter path of directory to store csv files: ")
    train_path = raw_input("Enter path of train csv file (Path upto p1_train.csv): ")
    slash = "/"
    # raw_path_string = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\train'
    # csv_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\ResultsCenter'
    # train_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\p1_train_pop_lab.csv'
#     colnames1 = ['TPM', 'Length']
#     # make csv files from quant.sf files
#     make_csv.make_csv_files(raw_path_string + slash, csv_path, slash,['Name'] + colnames1)
#
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
#
#     # create dataframe from all csv files
#     # each row corresponds to one accession number and the columns are TPM values of each transcript
#
#     # Reading the data from csv files and creating a data list of acession number tpm and length
#     files = listdir(csv_path)
#     for file in files:
#         name = file.split('.')[0]
#         data = pd.read_csv(csv_path + slash + file, usecols=colnames1, converters={'TPM': float,'Length':float})
#         data_list = [file] + data.TPM.tolist()+ data.Length.tolist()
#     #Create label popluation and sequence center
#         classifier_population = label_dict[name][0]
#         classifier_sequence_center = label_dict[name][1]
#         data_list = data_list + [classifier_population, classifier_sequence_center]
#         classifier_input.append(data_list)
#
# print "Read all csv files, creating dataframe"
# print datetime.datetime.now()
#
# # created dataframe will have following format
#
# # Name       TPM_1  TPM_2  TPM_3  TPM_4 ....  TPM_199324  label
# # ERR188021  value  value  value  value ....     value     TSI
# # ERR188022    .      .      .      .   ....       .       CEU
# #   .          .      .      .      .   ....       .        .
# #   .          .      .      .      .   ....       .        .
# #   .          .      .      .      .   ....       .        .
#
# df = pd.DataFrame(classifier_input)



# print "Created dataframe"
# print datetime.datetime.now()

df = eq_classes(raw_path_string, csv_path, train_path, slash, 'both')
prediction_with_pca(df)

# execute classifier
# prediction_with_pca(df)
# prediction_with_tree_classifier(df)  # 0 for predicting population
# prediction_with_tree_classifier(df, 1) # 1 for predicting sequence center
# prediction_with_pca(df)
# prediction_with_Kbest(df)
print datetime.datetime.now()