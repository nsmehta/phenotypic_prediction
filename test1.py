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
    raw_path=['/home/rasika/Documents/Computational Biology/Project/']
    output_path='/home/rasika/Documents/Computational Biology/Project/Results'
    colnames = ['Name', 'Length', 'EffectiveLength', 'TPM', 'NumReads']
    colnames1 = ['Name', 'TPM']
    colnames0 = ['Name', 'TPM']
    result =[]
    f1=pd.read_csv('/home/rasika/Documents/Computational Biology/Project/Data/Train/ERR188021/bias/quant.sf', sep='\t',header = 0,dtype='unicode')
    f1.to_csv('/home/rasika/Documents/Computational Biology/Project/Results/Resource.csv', index=False)
    metadata= pd.read_csv('/home/rasika/Documents/Computational Biology/Project/Results/Resource.csv', usecols=colnames0,converters={'Name':str, 'TPM':float})
    #metadata = metadata.rename(columns={'TPM': 'TPM_1'})
    print metadata.head()

    for path in raw_path:
        for roots,dir,files in os.walk(path):
            #print ("Files in: " + roots[:])
            directories = roots.split("/")
            directory_name = directories[-1]
            output_filename = directories[-2]
            #print directory_name
            if directory_name == "bias":
                for filename in files:
                    #print filename

                    if filename == 'quant.sf':
                        data = pd.read_csv(roots+'/'+filename, sep='\t',header = 0,dtype='unicode')
                        data.to_csv(output_path+'/'+output_filename+'.csv', index=False)

                        data = pd.read_csv(output_path+'/'+output_filename+'.csv', usecols=colnames1,converters={'Name':str,'TPM':float})

    data = data.merge(metadata, on='Name', how='inner')
    print data.head()
    print metadata.head()