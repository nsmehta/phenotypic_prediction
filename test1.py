import re
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os
from scipy.cluster.hierarchy import ward, dendrogram,linkage,fcluster,cophenet,distance
import scipy.cluster.hierarchy as hier
if __name__=='__main__':
    raw_path=['C:\Users\komal\pyex\CB']
    output_path='C:\Users\komal\pyex\CB\Results'
for path in raw_path:
    for roots,dir,files in os.walk(path):
        #print ("Files in: " + roots[:])
        directories = roots.split("\\")
        directory_name = directories[-1]
        output_filename = directories[-2]
        #print directory_name
        if directory_name == "no_bias":
            for filename in files:
                #print filename

                if filename == 'quant.sf':
                    data = pd.read_csv(roots+'\\'+filename, sep='\t',header = None,dtype='unicode')
                    data.to_csv(output_path+'\\'+output_filename+'.csv', index=False)
                    #with codecs.open(roots+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:
                    #    text=file_name.read()
