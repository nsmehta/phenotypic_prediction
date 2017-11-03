import re
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from sklearn.feature_extraction import DictVectorizer
import os
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

if __name__=='__main__':
    colnames1 = ['TPM']

    classifier_input = list()

    raw_path_string = raw_input("Enter Raw Path:")
    output_path = raw_input("Enter Output Path:")
    train_path = raw_input("Enter path of training data:")
    print train_path
    split_on_string = "/"
    raw_path = [raw_path_string]

    # raw_path = ['C:\\Users\\aditi\\Desktop\\Comp Bio\\train']
    # output_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\Results'
    # train_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\p1_train.csv'
    train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')

    for path in raw_path:
        for roots,dir,files in os.walk(path):
            directories = roots.split(split_on_string)
            directory_name = directories[-1]
            output_filename = directories[-2]
            if directory_name == "bias":
                for filename in files:
                    if filename == 'quant.sf':
                        data = pd.read_csv(roots + split_on_string + filename, sep='\t', header=0, dtype='unicode')
                        data.to_csv(output_path + split_on_string + output_filename + '.csv', index=False)
                        data = pd.read_csv(output_path + split_on_string + output_filename + '.csv', usecols=colnames1,converters={'TPM': float})
                        data_list = [output_filename] + data.TPM.tolist()
                        classifier_label =train_data[train_data['accession'].isin([output_filename])]["label"].tolist()[0]
                        data_list = data_list + [classifier_label]
                        classifier_input.append(data_list[:])


df = pd.DataFrame(classifier_input)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:-1], df.iloc[:, -1], test_size = 0.30, random_state = 0)
clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators= 10, max_features='auto', criterion="gini")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print "F1 score",f1_score(y_test,preds ,average='macro')
scores=cross_val_score(clf,df.iloc[:, 1:-1], df.iloc[:, -1],cv=5,scoring='f1_macro')
print "score with cross validation", scores
print " mean score", scores.mean()
print "accuracy score without k-fold", accuracy_score(preds,y_test)