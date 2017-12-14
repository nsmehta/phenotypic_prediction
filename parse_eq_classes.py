from os import listdir
import pandas as pd
import datetime

def get_features(raw_path_string, slash):
	accession_numbers = {}
	path = raw_path_string
	input_files = listdir(path)

	for file in input_files:
		input_file = open(path + slash + file + '/bias/aux_info/eq_classes.txt', 'r')
		lines = input_file.readlines()

		# w_file = open('/home/rasika/Desktop/' + file + '.txt', 'w')

		transcripts = lines[:199327]
		eq_classes = lines[199327:]

		unique_transcripts = {}

        # initializing
		for index in range(0, 199327):
			unique_transcripts[transcripts[index].strip('\r\n ')] = 0

		for entry in eq_classes:
			contents = entry.split('\t')
			index = int(contents[1].strip('\r\n')) - 1
			num_reads = int(contents[2].strip('\r\n'))

			if int(contents[0]) == 1:
				unique_transcripts[transcripts[index].strip('\r\n ')] += num_reads

		accession_numbers[file] = unique_transcripts
		# w_file.write(str(unique_transcripts))

	return accession_numbers


def create_dataframe_with_eq_class(raw_path_string, csv_path, train_path, slash, label_dict):

	df = None

	a_file = open('/home/rasika/Documents/Computational Biology/Project/Features/dataframe.csv', 'r')

	# test = open('/home/rasika/Documents/Computational Biology/Project/Features/test.txt', 'w')
	if not len(a_file.readlines()) > 0:
		classifier_input = list()
		col_names = ['Name', 'TPM', 'Length', 'EffectiveLength', 'NumReads']

	    # get the parsed equivalence classes data
		accession_numbers = get_features(raw_path_string, slash)
		print datetime.datetime.now()

		seen = False
		files = listdir(csv_path)
		for file in files:
			data_list = list()
			name = file.split('.')[0]
			data = pd.read_csv(csv_path + slash + file, usecols = col_names, converters = {'TPM': float, 'Count': float, 'Length': float, 'EffectiveLength': float, 'NumReads': float})

			unique_transcripts = accession_numbers[name]

			df = pd.DataFrame(list(unique_transcripts.iteritems()), columns = ['Name', 'Count'])
			data = data.merge(df, on='Name')

			data_list = [name] + data.Count.tolist() + data.TPM.tolist() + data.Length.tolist()

			classifier_population = label_dict[name][0]
			classifier_sequence_center = label_dict[name][1]

			data_list = data_list + [classifier_population, classifier_sequence_center, classifier_population + '-' + classifier_sequence_center]
			test.write(str(data_list))


			classifier_input.append(data_list)

		print "Read all csv files, creating dataframe"
		print datetime.datetime.now()

		df = pd.DataFrame(classifier_input)

		df.to_csv('/home/rasika/Documents/Computational Biology/Project/Features/dataframe.csv', index=False)
	else:
		df = pd.read_csv('/home/rasika/Documents/Computational Biology/Project/Features/dataframe.csv', sep=',', header=0, dtype='unicode')
	return df