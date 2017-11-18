import numpy as np
import csv
from kfold_cross_validation import run_feature_selection_and_classification
from matplotlib import pyplot
from os import path
import os

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

def classify_test_users(train_file, test_file, methods_to_run, test_file_to_get_users, options, class_1_weight = 1):
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    labels = np.concatenate((train_labels, test_labels))
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, (train_size + test_size))
    error, recall, precision, specificity = run_feature_selection_and_classification(methods_to_run, train_data, test_data, labels,
                                             train_index, test_index, test_user_ids,
                                             options, test_header, test_file_to_get_users, class_1_weight)
    print 'Error: %0.3f, Recall: %0.3f\nPrecision: %0.3f, Specificity: %0.7f' %(error, recall, precision, specificity)

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
        specificities = np.zeros(len(different_number_of_samples))
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
            if len(options) < 6:
                error, recall, precision, specificity = run_feature_selection_and_classification(methods_to_run, train_data_subset,
                                                                                                 test_data, labels, train_index,
                                                                                                 test_index, test_user_ids, options,
                                                                                                 test_header, test_file_to_get_users)
            else:
                error, recall, precision, specificity = run_feature_selection_and_classification(methods_to_run, train_data_subset,
                                                                                                 test_data, labels, train_index,
                                                                                                 test_index, test_user_ids, options,
                                                                                                 test_header, test_file_to_get_users,
                                                                                                 options[5])
            precisions[index] = precision
            recalls[index] = recall
            specificities[index] = specificity
            index = index + 1
        pyplot.figure()
        pyplot.ylim((-0.01, 1.1))
        pyplot.xlabel('Number of Samples')
        pyplot.plot(different_number_of_samples, precisions, color = 'r')
        pyplot.plot(different_number_of_samples, recalls, color = 'b')
        pyplot.plot(different_number_of_samples, specificities, color = 'g')
        if len(options) < 6:
            pyplot.legend([methods_to_run[1] + ' ' + options[2] + ' Precision',
                           methods_to_run[1] + ' ' + options[2]+ ' Recall',
                           methods_to_run[1] + ' ' + options[2] + ' Specificity'])
            pyplot.savefig(path.join(output_folder, 'different_samples_'+options[2]+'_'+methods_to_run[1]+'.png'))
        else:
            pyplot.legend([methods_to_run[1] + ' ' + options[2] + ' With Class Weight Precision',
                           methods_to_run[1] + ' ' + options[2]+ ' With Class Weight Recall',
                           methods_to_run[1] + ' ' + options[2] + ' With Class Weight Specificity'])
            pyplot.savefig(path.join(output_folder, 'different_samples_'+options[2]+'_'+methods_to_run[1]+'_with_class_weights.png'))            
    pyplot.close()


def test_different_subsets(train_folder, test_file, methods_to_run_list, test_file_to_get_users,
                           options_list, output_folder, train_files_count):
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    option_index = 0
    for methods_to_run in methods_to_run_list:
        print(methods_to_run)
        options = options_list[option_index]
        option_index = option_index + 1
        precisions = np.zeros(train_files_count)
        recalls = np.zeros(train_files_count)
        specificities = np.zeros(train_files_count)
        index = 0
        for file_in_train_folder in os.listdir(train_folder):
            if 'train' in file_in_train_folder: # As the folder contain train and test files.
                print(file_in_train_folder)
                train_data, train_header, train_labels, train_user_ids = get_data(path.join(train_folder,
                                                                                            file_in_train_folder))
                labels = np.concatenate((train_labels, test_labels))
                train_size = len(train_labels)
                test_size = len(test_labels)
                train_index = range(train_size)
                test_index = range(train_size, (train_size + test_size))
                if len(options) < 6:
                    error, recall, precision, specificity = run_feature_selection_and_classification(methods_to_run, train_data,
                                                                                                     test_data, labels, train_index,
                                                                                                     test_index, test_user_ids, options,
                                                                                                     test_header, test_file_to_get_users)
                else:
                    error, recall, precision, specificity = run_feature_selection_and_classification(methods_to_run, train_data,
                                                                                                     test_data, labels, train_index,
                                                                                                     test_index, test_user_ids, options,
                                                                                                     test_header, test_file_to_get_users,
                                                                                                     options[5])
                precisions[index] = precision
                recalls[index] = recall
                specificities[index] = specificity
                index = index + 1
        pyplot.figure()
        pyplot.ylim((-0.01, 1.1))
        pyplot.xlabel('Train Subset Number')
        index = np.arange(train_files_count)
        bar_width = 0.2
        pyplot.bar(index, precisions, bar_width, color = 'r')
        pyplot.bar(index + bar_width, recalls, bar_width, color = 'b')
        pyplot.bar(index - bar_width, specificities, bar_width, color = 'g')
        pyplot.xticks(index + bar_width, tuple([str(i) for i in range(train_files_count)]))

        if len(options) < 6:
            pyplot.legend([methods_to_run[1] + ' ' + options[2] + ' Precision',
                           methods_to_run[1] + ' ' + options[2]+ ' Recall',
                           methods_to_run[1] + ' ' + options[2] + ' Specificity'])
            pyplot.savefig(path.join(output_folder, 'different_subsets_'+options[2]+'_'+methods_to_run[1]+'.png'))
        else:
            pyplot.legend([methods_to_run[1] + ' ' + options[2] + ' With Class Weight Precision',
                           methods_to_run[1] + ' ' + options[2]+ ' With Class Weight Recall',
                           methods_to_run[1] + ' ' + options[2] + ' With Class Weight Specificity'])
            pyplot.savefig(path.join(output_folder, 'different_subsets_'+options[2]+'_'+methods_to_run[1]+'_with_class_weights.png'))            
    pyplot.close()


if __name__ == '__main__':
    train_file = 'train_subsets/0_train.csv'
    test_file = 'test_subsets/0_test.csv'
    
    number_of_features_to_select = 7
    
    # methods_to_run = ['linear_SVC', 'kernel_svm']
    # C = 40
    # class_1_weight = 10
    # kernel = 'rbf'

    # write_prediction = 0
    # prediction_file = ''
    # sparsity_param = 0.002
    # options = [sparsity_param, C, kernel, write_prediction, prediction_file]

    write_prediction = 0
    prediction_file = ''
    methods_to_run = ['linear_SVC', 'preceptron']
    options = [number_of_features_to_select, [100] * 10, '', write_prediction, prediction_file]
    classify_test_users(train_file, test_file, methods_to_run, '', options)
