import re
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from sklearn.feature_extraction import DictVectorizer
import os
#import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram,linkage,fcluster,cophenet,distance
import scipy.cluster.hierarchy as hier

if __name__=='__main__':
    labels = {}

    raw_path=['/home/rasika/Documents/Computational Biology/Project/']
    output_path='/home/rasika/Documents/Computational Biology/Project/Results'
    colnames1 = ['TPM']
    result =[]

    merged_df = pd.DataFrame()

    labels_data = pd.read_csv(raw_path[0] +'/' + 'p1_train.csv', sep='\t',header = 0,dtype='unicode')

    for row in labels_data.itertuples():
        key = row[1].split(',')[0]
        value = row[1].split(',')[1]
        labels[key] = value
    

    for path in raw_path:
        for roots,dir,files in os.walk(path):
            directories = roots.split("/")
            directory_name = directories[-1]
            output_filename = directories[-2]
            if directory_name == "bias":
                for filename in files:

                    if filename == 'quant.sf':
                        data = pd.read_csv(roots+'/'+filename, sep='\t',header = 0,dtype='unicode')
                        data.to_csv(output_path+'/'+output_filename+'.csv', index=False)
                        data = pd.read_csv(output_path+'/'+output_filename+'.csv', usecols=colnames1,converters={'TPM':float})
                        data['label'] = labels[output_filename]

                        merged_df = merged_df.append(data)

    print merged_df.head()