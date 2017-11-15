import numpy as np
import csv
from kfold_cross_validation import run_feature_selection_and_classification
from matplotlib import pyplot
from os import path

def get_data(data_file):
    csvfile = open(data_file,'rb')
    lines = csv.reader(csvfile, delimiter = ',')
    lines = list(lines)
    header = lines[0]
    # Remove headers from the data.
    del lines[0]
    data = np.array([[float(elem) for elem in line[2:]] for line in lines])
    labels = np.array([[int(elem) for elem in line[1]] for line in lines])
    user_ids = np.array([[elem for elem in line[0]] for line in lines])
    return data, header, labels, user_ids

def classify_test_users(train_file, test_file, methods_to_run, test_file_to_get_users, options):
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    labels = np.concatenate((train_labels, test_labels))
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, (train_size + test_size))
    run_feature_selection_and_classification(methods_to_run, train_data, test_data, labels,
                                             train_index, test_index, test_user_ids,
                                             options, test_header, test_file_to_get_users)

def test_different_number_of_samples(train_file, test_file, methods_to_run_list, test_file_to_get_users, options_list,
                                     different_number_of_samples, output_folder):
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    option_index = 0
    for methods_to_run in methods_to_run_list:
        print(methods_to_run)
        options = options_list[option_index]
        option_index = option_index + 1
        precisions = np.zeros(len(different_number_of_samples))
        recalls = np.zeros(len(different_number_of_samples))
        index = 0
        for samples_number in different_number_of_samples:
            print(samples_number)
            train_data_subset = train_data[:samples_number,]
            train_labels_subset = train_labels[:samples_number,]
            labels = np.concatenate((train_labels_subset, test_labels))
            train_size = len(train_labels_subset)
            test_size = len(test_labels)
            train_index = range(train_size)
            test_index = range(train_size, (train_size + test_size))
            error, recall, precision, specificity = run_feature_selection_and_classification(methods_to_run, train_data_subset, test_data, labels,
                                                                                             train_index, test_index, test_user_ids,
                                                                                             options, test_header, test_file_to_get_users)
            precisions[index] = precision
            recalls[index] = recall
            index = index + 1
            print(precision)
            print(recall)
        pyplot.figure()
        pyplot.xlabel('Number of Samples')
        pyplot.plot(different_number_of_samples, precisions, color = 'r')
        pyplot.plot(different_number_of_samples, recalls, color = 'b')
        pyplot.legend([methods_to_run[1] + ' Precision',  methods_to_run[1] + ' Recall'])
        pyplot.savefig(path.join(output_folder, 'different_samples_'+methods_to_run[1]+'.png'))
    pyplot.close()
    
