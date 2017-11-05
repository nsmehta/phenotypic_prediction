import os
import pandas as pd
from os import listdir

def make_csv_files(raw_path_string, csv_path):    
    print "Starting creating of csv files"

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    elif os.path.exists(csv_path) and len(list(csv_path)) > 0:
        print "Some csv files already present. Not creating again."
        return

    slash = "/"    

    columns = ['Name', 'TPM']
    input_files = listdir(raw_path_string)
    for file in input_files:
        data = pd.read_csv(raw_path_string + file + slash + 'bias' + slash + 'quant.sf', sep='\t', header=0, dtype='unicode')
        data = data[columns]
        data.to_csv(csv_path + slash + file + '.csv', header = columns, index=False)                
        
    print "Created csv files"