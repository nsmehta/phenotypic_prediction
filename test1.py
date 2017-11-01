import re
import codecs
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from sklearn.feature_extraction import DictVectorizer
import os
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster, cophenet, distance
import scipy.cluster.hierarchy as hier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

if __name__ == '__main__':
    # raw_path=['C:\\Users\\aditi\\Desktop\\Comp Bio\\train']
    # output_path='C:\\Users\\aditi\\Desktop\\Comp Bio\\Results'
    # train_path='C:\\Users\\aditi\\Desktop\\Comp Bio\\p1_train.csv'

    raw_path = ['C:\\Users\\aditi\\Desktop\\Comp Bio\\test']
    output_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\test result'
    train_path = 'C:\\Users\\aditi\\Desktop\\Comp Bio\\p1_train.csv'
    colnames1 = ['TPM']
    orig_stdout=sys.stdout
    f=open('output.txt','w')
    sys.stdout=f
    actual_label=[]
    pred_label=[]
    classifier_input = list()

    train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')

    for path in raw_path:
        for roots, dir, files in os.walk(path):
            directories = roots.split("\\")
            directory_name = directories[-1]
            output_filename = directories[-2]

            if directory_name == "bias":
                for filename in files:
                    if filename == 'quant.sf':
                        data = pd.read_csv(roots + '\\' + filename, sep='\t', header=0, dtype='unicode')
                        data.to_csv(output_path + '\\' + output_filename + '.csv', index=False)
                        data = pd.read_csv(output_path + '\\' + output_filename + '.csv', usecols=colnames1,
                                           converters={'TPM': float})
                        data_list = [output_filename] + data.TPM.tolist()
                        # name_list = data.Name.tolist()
                        classifier_label =train_data[train_data['accession'].isin([output_filename])]["label"].tolist()[0]
                        data_list = data_list + [classifier_label]
                        classifier_input.append(data_list[:])
                        # classifier_input.append()

                        # print "\n".join(map(str, classifier_input))
# index = 0
# for name in name_list:
#     name += '_' + str(index)
#     index += 1

# name_list.append('label')

# print len(name_list)

# df = pd.DataFrame(classifier_input, name_list)
df = pd.DataFrame(classifier_input)
print df

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# print len(train), len(test)
# print test

features = df.columns[1:199325]

# print features

y = pd.factorize(train[199325])[0]

# print y

clf = RandomForestClassifier(n_jobs=2, random_state=0)

# print train[features]

clf.fit(train[features], y)

for index, row in test.iterrows():
    # actual_label.append(row[-2])
    print "Actual label", row[-2]

for pred_val in clf.predict(test[features]):
    # pred_label.append(pred_val)
    print "Pred label",pred_val


# print clf.predict(test[features])[0]

# print actual_label
# print pred_label
sys.stdout=orig_stdout
f.close()
