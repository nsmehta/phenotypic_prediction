from os import listdir

def get_features(raw_path_string):
	accession_numbers = {}

	# path = '/home/rasika/Documents/Computational Biology/Project/Data/'
	# path = raw_input("Enter path where data is located (Location of accession number dirs): ")
	path = raw_path_string
	input_files = listdir(path)
	# slash = '/'
	slash = '\\'
	# ERR188021/bias/aux_info/eq_classes.txt

	for file in input_files:
		input_file = open(path + slash + file + '/bias/aux_info/eq_classes.txt', 'r')
		lines = input_file.readlines()

		# w_file = open('F:\\Computational Biology\\project\\train test\\' + file + '\\unique_transcripts.txt', 'w')

		transcripts = lines[:199327]
		eq_classes = lines[199327:]
from os import listdir

def get_features(raw_path_string):
	accession_numbers = {}

	# path = '/home/rasika/Documents/Computational Biology/Project/Data/'
	# path = raw_input("Enter path where data is located (Location of accession number dirs): ")
	path = raw_path_string
	input_files = listdir(path)
	# slash = '/'
	slash = '\\'
	# ERR188021/bias/aux_info/eq_classes.txt

	for file in input_files:
		input_file = open(path + slash + file + '/bias/aux_info/eq_classes.txt', 'r')
		lines = input_file.readlines()

		# w_file = open('F:\\Computational Biology\\project\\train test\\' + file + '\\unique_transcripts.txt', 'w')

		transcripts = lines[:199327]
		eq_classes = lines[199327:]

		unique_transcripts = {}

		seen = False

        # initializing
		for index in range(0, 199327):
			unique_transcripts[transcripts[index].strip('\r\n ')] = 0

		for entry in eq_classes:
			# if not seen:
			contents = entry.split('\t')
			index = int(contents[1].strip('\r\n')) - 1
			num_reads = int(contents[2].strip('\r\n'))
			# print contents
			if int(contents[0]) == 1:
				seen = True
				unique_transcripts[transcripts[index].strip('\r\n ')] += num_reads
			# elif int(contents[0]) > 1:
			# 	unique_transcripts[transcripts[index]] = 0

				# print contents
				# print unique_transcripts
		# w_file.write(str(unique_transcripts))
		accession_numbers[file] = unique_transcripts

	# print accession_numbers[file]
	return accession_numbers