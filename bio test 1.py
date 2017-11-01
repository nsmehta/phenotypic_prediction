import re
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from sklearn.feature_extraction import DictVectorizer
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram,linkage,fcluster,cophenet,distance
import scipy.cluster.hierarchy as hier
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate

if __name__=='__main__':
    raw_path=['C:\\Users\\aditi\\Desktop\\Comp Bio\\train']
    output_path='C:\\Users\\aditi\\Desktop\\Comp Bio\\Results'
    train_path='C:\\Users\\aditi\\Desktop\\Comp Bio\\p1_train.csv'
    # colnames1 = ['Name', 'TPM']
    colnames1 = ['TPM']
    # f1=pd.read_csv('C:\\Users\\aditi\\Desktop\\Comp Bio\\train\\ERR188021\\bias\\quant.sf', sep='\t',header = 0,dtype='unicode')
    # f1.to_csv('C:\\Users\\aditi\\Desktop\\Comp Bio\\test result\\Resource.csv', index=False)
    # metadata= pd.read_csv('C:\\Users\\aditi\\Desktop\\Comp Bio\\test result\\Resource.csv', usecols=colnames1,converters={'Name':str, 'Length':float,'TPM':float})

    classifier_input = list()

    train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')

    for path in raw_path:
        for roots,dir,files in os.walk(path):
            #print ("Files in: " + roots[:])
            directories = roots.split("\\")
            directory_name = directories[-1]
            output_filename = directories[-2]

            #print directory_name
            if directory_name == "bias":
                for filename in files:
                    if filename == 'quant.sf':
                        data = pd.read_csv(roots+'\\'+filename, sep='\t',header = 0,dtype='unicode')
                        data.to_csv(output_path+'\\'+output_filename+'.csv', index=False)
                        data = pd.read_csv(output_path+'\\'+output_filename+'.csv', usecols=colnames1,converters={'TPM':float})
                        data_list = [output_filename] + data.TPM.tolist()
                        classifier_label = train_data[train_data['accession'].isin([output_filename])]["label"].tolist()[0]
                        data_list = data_list + [classifier_label]
                        classifier_input.append(data_list[:])
                        # classifier_input.append()


    # print "\n".join(map(str, classifier_input))
    print pd.DataFrame(classifier_input)
    # data = data.merge(metadata, on='Name', how='inner')
    # from sklearn.ensemble import RandomForestClassifier
    #
    # print data.head()