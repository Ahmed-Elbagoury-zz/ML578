import math
import csv
from my_PCA import MyPCA
import numpy as np
from matplotlib import pyplot
import os.path as path
def kfold_cross_validation(k, train_file, methods_to_run,
                           output_folder, options):
    train_csvfile = open(train_file,'rb')
    lines = csv.reader(train_csvfile, delimiter = ',')
    lines = list(lines)
    header = lines[0]
    # Remove headers from the data.
    del lines[0]
    samples_number = len(lines)
    # Transform it to matrix.
    # Skip the user id and the label columns.
    data = np.array([[float(elem) for elem in line[2:]] for line in lines])
    labels = np.array([[int(elem) for elem in line[1]] for line in lines])
    features_number = len(data[0])
    for i in range(k):
        start_i = int(math.floor(samples_number * i / k))
        end_i = int(math.floor(samples_number * (i + 1) / k))
        validation_index = range(start_i, end_i)
        points_index = range(samples_number)
        train_index = list(set(points_index).difference(set(validation_index)));
        train_labels = labels[train_index, :]
        zero_indices = [j for j in range(len(train_labels)) if train_labels[j] == 0]
        one_indices = [j for j in range(len(train_labels)) if train_labels[j] == 1]
        if 'features_histogram' in methods_to_run:
            # Caclulate features histogram for each fold training data.
            for feature in range(features_number):
                train_data = data[train_index, feature]
                feature_column_class0 = train_data[zero_indices]
                feature_column_class1 = train_data[one_indices]
                pyplot.figure()
                pyplot.hist(feature_column_class0, 100, label='Class 0')
                pyplot.hist(feature_column_class1, 100, label='Class 1')
                pyplot.legend(loc='upper right')
                pyplot.savefig(path.join(output_folder, 'hist_fold=' + str(i) +
                                         '_' + header[feature+2] + '.png'))
                pyplot.close()
        if 'pca' in methods_to_run:
            # Run PCA.
            myPca = MyPCA()
            train_data = data[train_index, :]
            #print(train_data[5566, :])
            #print(train_data[8660, :])
            #train_data = np.delete(train_data, [2695,5789,24610], 0)
            #train_labels = np.delete(train_labels, [2695,5789,24610], 0)
            zero_indices = [j for j in range(len(train_labels)) if train_labels[j] == 0]
            one_indices = [j for j in range(len(train_labels)) if train_labels[j] == 1]
            myPca.fit(train_data, n_components = 2)
            # Visualize the training data.
            projected_data = myPca.transform(train_data)
            pyplot.figure()
            #projected_data_np = np.array(projected_data[zero_indices, 0])
            #points_index_np = np.array(range(len(projected_data_np)))
            #print(points_index_np[projected_data_np >= 3e9])
            pyplot.plot(projected_data[zero_indices, 0], projected_data[zero_indices, 1],
                        'o', color = 'r')
            pyplot.plot(projected_data[one_indices, 0], projected_data[one_indices, 1],
                        'x', color = 'b')
            pyplot.savefig(path.join(output_folder, 'pca_fold=' +
                                     str(i) + '.png'))
            pyplot.close()
        if 'f1' in methods_to_run: # Run feature selection
            print('f1')
        if 'f2' in methods_to_run: # Run feature selection
            print('f2')
        if 'svm' in methods_to_run: # Run SVM.
            print('svm')
        if 'preceptron' in methods_to_run: # Run multilayer preceptron.
            print('preceptron')
