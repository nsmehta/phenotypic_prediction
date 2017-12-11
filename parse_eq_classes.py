from os import listdir

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