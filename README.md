# phenotypic_prediction
Phenotypic Prediction from Transcriptomic Features 

Our project aims to predict the population and sequence center to which a sample belongs from the output of Salmon.

How to run:
**python prediction.py \<address to the model dump\> \<address to the test samples root\> \<address to the created csv files\> \<address to the test labels file\> \<type of slash to use in paths\> \<flag to indicate whether to use equivalence classes\> <address to the csv file created using equivalence classes\>**

1. <address to the model dump>: Directory where the model dump will be saved. It is required that the directory contain 3 files named, 'population_model.sav', 'sc_model.sav', 'population_sc_model.sav'.
2. <address to the test samples root>: Directory containing all accession number folders of input
3. <address to the created csv files>: Directory where CSV files created from the input are stored. When the program is run the first time, these files will be created and used without creation later. In order to create the files again, an empty directory should be supplied as an argument.
4. <address to the test labels file>: Directory where the CSV containing labels is stored.
5. <type of slash to use in paths>: In the program paths are being formed by concatenation. This argument specifies whether to use a forward or backward slash in the paths.
6. <flag to indicate whether to use equivalence classes>: This flag is used to indicate if the feature from equivalnce classes should be used, they will be used if it is True.
7. <address to the csv file created using equivalence classes>: Directory where dump of dataframe created using feature from equivalence class is stored. Since the creation takes a long time, the created dataframe will be stored in this directory when it is used the first time. This directory requires a file named 'dataframe.csv' to present.
