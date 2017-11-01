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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

if __name__=='__main__':
    # raw_path=['C:\\Users\\aditi\\Desktop\\Comp Bio\\train']
    # output_path='C:\\Users\\aditi\\Desktop\\Comp Bio\\Results'
    # train_path='C:\\Users\\aditi\\Desktop\\Comp Bio\\p1_train.csv'

    raw_path=['/home/rasika/Documents/Computational Biology/Project/']
    output_path='/home/rasika/Documents/Computational Biology/Project/Results'

    colnames1 = ['TPM']    

    classifier_input = list()

    train_data = pd.read_csv(raw_path[0] +'/' + 'p1_train.csv', sep=',', header=0, dtype='unicode')

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
                        data_list = [output_filename] + data.TPM.tolist()
                        # name_list = data.Name.tolist()
                        classifier_label = train_data[train_data['accession'].isin([output_filename])]["label"].tolist()[0]
                        data_list = data_list + [classifier_label]
                        classifier_input.append(data_list[:])
                        # classifier_input.append()

    # print "\n".join(map(str, classifier_input))
# index = 0
# for name in name_list:
# 	name += '_' + str(index)
# 	index += 1

# name_list.append('label')

# print len(name_list)

# df = pd.DataFrame(classifier_input, name_list)
df = pd.DataFrame(classifier_input)

# df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
# print df

# train, test = df[df['is_train']==True], df[df['is_train']==False]
split=len(df)

print split

train, test = train_test_split(df, test_size=0.5)


print "lengths :: ", len(train), len(test)
print "train data\n", train, "\n"
print "test data\n", test, "\n"

print "labels ::: ", train[199325]

features = df.columns[1:199325]

y, label = pd.factorize(train[199325])

print "factorised :: ", y

clf = RandomForestClassifier(n_jobs=2, random_state=0)

# print "train features", train[features]

clf.fit(train[features], y)

# done = False
# for index, row in test.iterrows():
# 	if not done:
# 		print "Actual label", row[-1]
# 		break

# print clf.predict(test[features])[0]

actual_y = pd.factorize(test[199325])
print "actual :: ", actual_y[0]

preds = clf.predict(test[features])
print "predicted ", preds

# orig_labels = train[199325]
# print "orig_labels ", orig_labels
print "predicted label", label[preds]

print f1_score(actual_y[0], preds, average='weighted')

# for index in preds:
# 	print label[index]