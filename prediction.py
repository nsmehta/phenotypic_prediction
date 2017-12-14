import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
import make_csv
import datetime
from os import listdir
import parse_eq_classes
import pickle
import sys

def cross_val(df, folds, type, classifier):
# type 1 = pop + seq center
# type 2 = seq center
# type 3 = pop

# split the array into 'folds' partitions
	df = np.array_split(df, folds)
	f1_score1_p = []
	accuracy_p  = []
	f1_score1_c = []
	accuracy_c  = []
	f1_score = 0
	accuracy = 0

	for i in range(len(df)):
	    test = pd.DataFrame(df[i])

	    train = df[:i] + df[i + 1:]
	    train = pd.concat(train)

	    X_train = train.iloc[:, 1:-3]
	    y_train = train.iloc[:, -type]
	    X_test  = test.iloc[:, 1:-3]
	    y_test  = test.iloc[:, -type]

	    clf = None
	    if classifier == 0:
			clf = RandomForestClassifier(n_jobs = -1, random_state = 0, n_estimators = 10, max_features = None, criterion = "gini")
			y_train = pd.factorize(y_train)[0]
			y_test = pd.factorize(y_test)[0]
	    else:	    	
		    clf = tree.DecisionTreeClassifier(random_state = 0, max_features = None, criterion = 'gini', splitter = 'best',
		                                      max_depth = None, min_samples_split = 2, min_samples_leaf = 5, class_weight = 'balanced')

	    fit_model = clf.fit(X_train, y_train)
	    output_pred = fit_model.predict(X_test)

	    if type == 3: # population
			f1_score1_p.append(f1_score(y_test, output_pred, average='weighted'))
			accuracy_p.append(accuracy_score(y_test, output_pred))
			
	    elif type == 2: # seq center
			f1_score1_c.append(f1_score(y_test, output_pred, average='weighted'))
			accuracy_c.append(accuracy_score(y_test, output_pred))

	    elif type == 1: # population-seq center
			population_pred = []
			sequence_pred   = []
			population_test = []
			sequence_test   = []

			# print('y_test', y_test)
			# print('output_pred', output_pred)
			y_array = np.array(y_test)

			for index in range(len(output_pred)):
				p, sc = output_pred[index].split('-')
				p1, sc1 = y_array[index].split('-')

				population_pred.append(p)
				sequence_pred.append(sc)

				population_test.append(p1)
				sequence_test.append(sc1)
				
			f1_score1_c.append(f1_score(sequence_test, sequence_pred, average='weighted'))
			accuracy_c.append(accuracy_score(sequence_test, sequence_pred))
			
			f1_score1_p.append(f1_score(population_test, population_pred, average='weighted'))
			accuracy_p.append(accuracy_score(population_test, population_pred))

	if type == 3:
		f1 = np.mean(f1_score1_p)
		accuracy1 = np.mean(accuracy_p)		
		print "Mean F1 Score for Population", f1
		print "Mean accuracy for Population", accuracy1
		# w_file.write(str("Mean F1 Score for Population" + f1 + '\n'))
		# w_file.write(str("Mean accuracy for Population" + accuracy1 + '\n'))
		f1_score = f1
		accuracy = accuracy1


	elif type == 2:
		f1_c = np.mean(f1_score1_c)
		accuracy1_c = np.mean(accuracy_c)
		print "Mean F1 Score for sequence center", f1_c
		print "Mean accuracy for sequence center", accuracy1_c
		# w_file.write(str("Mean F1 Score for sequence center" + f1_c + '\n'))
		# w_file.write(str("Mean accuracy for sequence center" + accuracy1_c + '\n'))
		f1_score = f1_c
		accuracy = accuracy1_c

	elif type == 1:
		f1 = np.mean(f1_score1_p)
		accuracy1 = np.mean(accuracy_p)
		print "Mean F1 Score for Population", f1
		print "Mean accuracy for Population", accuracy1
		f1_c = np.mean(f1_score1_c)
		accuracy1_c = np.mean(accuracy_c)
		print "Mean F1 Score for sequence center", f1_c
		print "Mean accuracy sequence center", accuracy1_c

		# w_file.write(str("Mean F1 Score for Population" + repr(f1) + '\n'))
		# w_file.write(str("Mean accuracy for Population" + repr(accuracy1) + '\n'))
		# w_file.write(str("Mean F1 Score for sequence center" + repr(f1_c) + '\n'))
		# w_file.write(str("Mean accuracy sequence center" + repr(accuracy1_c) + '\n'))
		f1_score = f1_c
		accuracy = accuracy1_c

	return f1_score, accuracy1_c


def random_forest_classifier(X, y, df, column):
	clf = RandomForestClassifier(n_jobs = -1, random_state = 0, n_estimators = 10, max_features = None, criterion = "gini")

	# f1 score
	scores = cross_val_score(clf, X, y, cv = 5, scoring = 'f1_macro')
	print "F1 scores with 5 fold cross validation for Population RF ", scores
	print "F1 score", scores.mean()
	# w_file.write(str("F1 scores with 5 fold cross validation for Population RF " + scores + '\n'))
	# w_file.write(str("F1 score", scores.mean() + '\n'))

	# accuracy
	scores = cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy')
	print "accuracy scores with 5 fold cross validation for Population RF", scores
	print "mean of accuracy", scores.mean()
	# w_file.write(str("accuracy scores with 5 fold cross validation for Population RF" + scores + '\n'))
	# w_file.write(str("mean of accuracy" + repr(scores.mean()) + '\n'))

	cross_val(df, 5, column, 0)


def decision_tree_classifier(X, y, df, column):
	print('Starting decision tree classifier')
	# w_file.write(str('Decision tree classifier' + '\n'))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

	clf = tree.DecisionTreeClassifier(max_features = None, criterion = 'gini', splitter = 'best', max_depth = None, min_samples_split = 2, min_samples_leaf = 5, class_weight = 'balanced')
	fit_model = clf.fit(X_train, y_train)
	output_pred = fit_model.predict(X_test)

	print("F1 score predicted w/o cross val DT :: ", f1_score(y_test, output_pred, average='weighted'))
	# w_file.write(str("F1 score predicted w/o cross val DT :: "))
	# w_file.write("%s" % f1_score(y_test, output_pred, average='weighted'))
	# w_file.write('\n')

	# scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
	# print "F1 scores with 5 fold cross validation", scores
	# print "F1 score", scores.mean()

	# # accuracy
	# scores = cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy')
	# print "accuracy scores with 5 fold cross validation", scores
	# print "mean of accuracy", scores.mean()

	# b = np.array([y])
	# X = np.hstack((X,b.T))
	# df_new = pd.DataFrame(X)
	return cross_val(df, 5, column, 1) # 5 is the number of folds


def get_extra_trees_classifier(df, column):
	clf = ExtraTreesClassifier(random_state=0)
	y = df.iloc[:, -column]
	y_numerical = pd.factorize(y)[0]
	
	clf = clf.fit(df.iloc[:, 1:-3], y_numerical)
	model = SelectFromModel(clf, threshold="mean", prefit=True)
	X_new = model.transform(df.iloc[:, 1:-3])

	return X_new, y_numerical


# method for selecting features with extra trees classifier
def prediction_with_tree_classifier(df, column):
	# Creating a clf, create a data frame y of only labels and fit the model using dataframe leaving first and last column
	X_new, y_numerical = get_extra_trees_classifier(df, column)

	# perform classification using the selected features
	# random_forest_classifier(X_new, y_numerical, df, column)
	return decision_tree_classifier(df.iloc[:, 1:-3], y_numerical, df, column)


def predict_population(dataframe):
	print('---------------------F1 score and Accuracy for Population------------------------')
	# w_file.write(str('\n---------------------F1 score and Accuracy for Population------------------------' + '\n'))
	f1_score, accuracy = prediction_with_tree_classifier(dataframe, 3)


def predict_sequence_center(dataframe):
	print('---------------------F1 score and Accuracy for Sequence Center------------------------')
	# w_file.write(str('\n---------------------F1 score and Accuracy for Sequence Center------------------------' + '\n'))
	f1_score, accuracy = prediction_with_tree_classifier(dataframe, 2)


def predict_population_seq_center(dataframe):
	print('---------------------F1 score and Accuracy for Population and Sequence Center------------------------')
	# w_file.write(str('---------------------F1 score and Accuracy for Population and Sequence Center------------------------' + '\n'))
	f1_score, accuracy = prediction_with_tree_classifier(dataframe, 1)


def prediction_with_pca(df):
    sc = StandardScaler()
    X_std = sc.fit_transform(df.iloc[:, 1:-3])
    pca = PCA(n_components=0.9, svd_solver='full', random_state=0)
    # pca.fit(df.iloc[:, 1:-1])
    # y = df.iloc[:, -3:]
    # y_numerical = pd.factorize(y)[0]
    X_new = pca.fit_transform(X_std)
    # random_forest_classifier(X_new, df.iloc[:, -1])
    # decision_tree_classifier_multi(X_new, y)
    return X_new


# def predict_population_with_pca(df):
# 	X_new = prediction_with_pca(df)
# 	def predict_population(dataframe):
# 	print('---------------------F1 score and Accuracy for Population with PCA------------------------')
# 	w_file.write(str('\n---------------------F1 score and Accuracy for Population with PCA------------------------' + '\n'))
# 	decision_tree_classifier(X_new, y_numerical, df, 3)


def create_dataframe(files, features, col_names):
	classifier_input = list()

	for file in files:
		name = file.split('.')[0]
		data = pd.read_csv(csv_path + slash + file, usecols = col_names, converters = {'TPM': float, 'Length':float, 'EffectiveLength': float, 'NumReads': float})

		data_list = [name]
		for f in features:
			data_list.extend(data[f].tolist())
		
	    #Create label popluation and sequence center
		classifier_population = label_dict[name][0]
		classifier_sequence_center = label_dict[name][1]
		data_list = data_list + [classifier_population, classifier_sequence_center, classifier_population + '-' + classifier_sequence_center]
		classifier_input.append(data_list)

	df = pd.DataFrame(classifier_input)
	return df


def main():
	args = sys.argv

	model_dump_path = args[1]
	raw_path_string = args[2]
	csv_path = args[3]
	train_path = args[4]
	slash = args[5]
	eq_class = args[6]

	print('args', model_dump_path, raw_path_string, csv_path, train_path, slash)

	scores = {}
	df = None

	# w_file = open('/home/rasika/Documents/Computational Biology/Project/output_file.txt', 'a')
	# w_file.write('\n\n' + str(datetime.datetime.now()))
	col_names = ['TPM', 'Length', 'EffectiveLength', 'NumReads']

	# make csv files from quant.sf files
	make_csv.make_csv_files(raw_path_string + slash, csv_path, slash, ['Name'] + col_names)

	label_dict = {}
	# store the labels from train file in a dictionary
	train_data = pd.read_csv(train_path, sep=',', header=0, dtype='unicode')
	for i, row in train_data.iterrows():
		label_dict[row[0]] = (row[1], row[2])

	print "Started reading csv files"
	print datetime.datetime.now()

	# Reading the data from csv files and creating a data list of acession number, tpm, length and effective length
	files = listdir(csv_path)
	if not eq_class:
		df = create_dataframe(files, ['TPM', 'Length',], col_names)
	else:
		df = parse_eq_classes.create_dataframe_with_eq_class(raw_path_string, csv_path, train_path, slash, label_dict)
		print('back from creation')

	print(df.head())

	print "Read all csv files, created dataframe"
	print datetime.datetime.now()

	# created dataframe will have following format
	# Name       TPM_1  TPM_2  TPM_3  TPM_4 ....  TPM_199324  label
	# ERR188021  value  value  value  value ....     value     TSI
	# ERR188022    .      .      .      .   ....       .       CEU
	#   .          .      .      .      .   ....       .        .
	#   .          .      .      .      .   ....       .        .
	#   .          .      .      .      .   ....       .        .

	# w_file.write("\n Predicting for Population, Sequence Center and Both on full data: TPM, Length, Effective Length, NumReads\n")


	f1_score_p, accuracy_p = predict_population(df)
	scores['Population'] = (f1_score_p, accuracy_p)

	f1_score_sc, accuracy_sc = predict_sequence_center(df)
	scores['Sequence Center'] = (f1_score_sc, accuracy_sc)
	
	f1_score_p_sc, accuracy_p_sc = predict_population_seq_center(df)
	scores['Population and Sequence Center'] = (f1_score_p_sc, accuracy_p_sc)

	# predict_population_with_pca(df)
	# predict_sequence_center_with_pca(df)
	# predict_population_seq_center_with_pca(df)

scores = main()