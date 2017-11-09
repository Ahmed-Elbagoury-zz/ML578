import math
import csv
from my_PCA import MyPCA
import numpy as np
from matplotlib import pyplot
import os.path as path
from calculate_performance_measures import calculate_performance_measures
from svm import *
from fea_selection import *
import math

def write_predictions(validation_users_id, predicted_labels, options, test_file_to_get_users):
    write_prediction = options[3]
    if write_prediction == 1:
        test_csvfile = open(test_file_to_get_users,'rb')
        lines = csv.reader(test_csvfile, delimiter = ',')
        test_users = []
        for line in lines:
            test_users.append(line[0])
        prediction_filename = options[4]
        users_number = len(validation_users_id)
	validation_users_dict = {}
	index = 0
	for user in range(users_number):
            user_id = validation_users_id[user]
            validation_users_dict[''.join(user_id)] = index
            index = index + 1
        prediction_file = open(prediction_filename, 'w')
        prediction_file.write('msno,is_churn\n')
        for user in test_users:
            if user in validation_users_dict:
                normalized_prediction = 1.0/(1.0+ math.e **( -1 * predicted_labels[validation_users_dict[user]]))
                prediction_file.write(user+','+str(normalized_prediction)+'\n')
	    else:
                prediction_file.write(user+',0\n')
        prediction_file.close()

def run_feature_selection_and_classification(methods_to_run, train_data, validation_data, labels,
                                             train_index, validation_index, validation_users_id,
                                             options, header, test_file_to_get_users):
	if 'univariate_fea_selection' in methods_to_run:
		# Run feature selection.
		number_of_features_to_select = options[0]
		selected_features_indices = univariate_fea_selection(train_data, labels[train_index],
                                                                     number_of_features_to_select)
		train_data = train_data[:, selected_features_indices]
		validation_data = validation_data[:, selected_features_indices]
		selected_featuees = np.array(header)[np.array(selected_features_indices)+2]
		for selected_feature in range(len(selected_featuees)):
			print(selected_featuees[selected_feature])
	if 'linear_SVC' in methods_to_run:
		# Run feature selection.
		sparsity_param = options[0]
		selected_features_indices = my_SelectFromModel(train_data, labels[train_index],
                                                               sparsity_param)
		train_data = train_data[:, selected_features_indices]
		validation_data = validation_data[:, selected_features_indices]
		selected_featuees = np.array(header)[np.array(selected_features_indices)+2]
		for selected_feature in range(len(selected_featuees)):
			print(selected_featuees[selected_feature])
	if 'linear_svm' in methods_to_run: # Run linear SVM.
		C = options[1]
		linear_svm_model = train_linear_svm(train_data, labels[train_index], C)
		predicted_labels, predicted_values = classify(linear_svm_model, validation_data)
		write_predictions(validation_users_id, predicted_values, options, test_file_to_get_users)
		error, recall, precision, specificity = calculate_performance_measures(predicted_labels,
                                                                                       labels[validation_index])
                return error, recall, precision, specificity
	if 'kernel_svm' in methods_to_run: # Run Kernel SVM.
		C = options[1]
		kernel_svm_model = train_kernel_svm(train_data, labels[train_index], C)
		predicted_labels, predicted_values = classify(kernel_svm_model, validation_data)
		write_predictions(validation_users_id, predicted_values, options, test_file_to_get_users)
		error, recall, precision, specificity = calculate_performance_measures(predicted_labels,
                                                                                       labels[validation_index])
                return error, recall, precision, specificity
	if 'one_class_svm' in methods_to_run:
		temp = np.array([label[0] for label in labels[train_index]]) == 0
		train_data = train_data[temp, :]
		one_class_svm_model = train_one_class_svm(train_data, options[2])
		predicted_labels, predicted_values = classify_one_class_svm(one_class_svm_model, train_data)
		error, recall, precision, specificity = calculate_performance_measures(predicted_labels,
                                                                                       labels[validation_index])
		write_predictions(validation_users_id, predicted_values, options, test_file_to_get_users)    
                return error, recall, precision, specificity
	if 'preceptron' in methods_to_run: # Run multilayer preceptron.
		print('preceptron')
	return 0, 0, 0, 0

from scipy import stats
import scipy as sp
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
    user_ids = np.array([[elem for elem in line[0]] for line in lines])
    features_number = len(data[0])
    measures = np.zeros([4, k])
    for i in range(k):
        start_i = int(math.floor(samples_number * i / k))
        end_i = int(math.floor(samples_number * (i + 1) / k))
        validation_index = range(start_i, end_i)
        points_index = range(samples_number)
        train_index = list(set(points_index).difference(set(validation_index)));
        train_labels = labels[train_index, :]
        zero_indices = [j for j in range(len(train_labels)) if train_labels[j] == 0]
        one_indices = [j for j in range(len(train_labels)) if train_labels[j] == 1]
        train_data = data[train_index, :]
        validation_data = data[validation_index, :]
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
            myPca.fit(train_data, n_components = 2)
            # Visualize the training data.
            projected_data = myPca.transform(train_data)
            pyplot.figure()
            pyplot.plot(projected_data[zero_indices, 0], projected_data[zero_indices, 1],
                        'o', color = 'r')
            pyplot.plot(projected_data[one_indices, 0], projected_data[one_indices, 1],
                        'x', color = 'b')
            pyplot.savefig(path.join(output_folder, 'pca_fold=' +
                                     str(i) + '.png'))
            pyplot.close()
        error, recall, precision, specificity = run_feature_selection_and_classification(methods_to_run,
                                                                                         train_data, validation_data,
                                                                                         labels, train_index, validation_index,
                                                                                         user_ids[validation_index, :], options,
                                                                                         header, '')
        measures[0, i] = error
	measures[1, i] = recall
	measures[2, i] = precision
	measures[3, i] = specificity
    stats_vals = []
    alpha_val = 0.95
    for i in range(4):
        mean, error = mean_confidence_interval(measures[i, :], alpha_val)
        stats_vals.append([mean, error])
    return stats_vals

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h
