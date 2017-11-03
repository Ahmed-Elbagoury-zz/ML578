import csv
import numpy as np
from matplotlib import pyplot
import os.path as path
def generate_histograms(train_file, output_folder):
    train_csvfile = open(train_file,'rb')
    lines = csv.reader(train_csvfile, delimiter = ',')
    lines = list(lines)
    # Remove headers.
    del lines[0]
    data = np.array(lines)
    # Split data into positive class and neg class.
    data_class1 = data[data[:,1] == '0']
    data_class2 = data[data[:,1] == '1']
    features_number = len(data[0])
    for feature in range(features_number):
        if feature == 0 or feature == 1:
            # Skip user id and class label.
            continue
        feature_column_class1 = data_class1[:,feature]
        feature_column_class2 = data_class2[:,feature]
        feature_column_class1 = feature_column_class1.astype(np.float)
        feature_column_class2 = feature_column_class2.astype(np.float)
        pyplot.figure()
        pyplot.hist(feature_column_class1, 100, label='Class 0')
        pyplot.hist(feature_column_class2, 100, label='Class 1')
        pyplot.legend(loc='upper right')
        pyplot.savefig(path.join(output_folder, str(feature) + '.png'))

