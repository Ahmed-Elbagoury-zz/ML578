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

def test_different_thresholds(train_file, test_file, methods_to_run_list, test_file_to_get_users, options_list, threshold_list, output_folder, class_1_weight = 1):
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, train_size + test_size)
    labels = np.concatenate((train_labels, test_labels))
    option_index = 0
    for method_to_run in methods_to_run_list:
        print(method_to_run)
        options = options_list[option_index]
        option_index = option_index + 1
        precisions = np.zeros(len(threshold_list))
        recalls = np.zeros(len(threshold_list))
        specificities = np.zeros(len(threshold_list))
        index = 0
        if len(options) < 6:
          error, recall, precision, specificity = run_feature_selection_and_classification(method_to_run, train_data, test_data, labels, train_index, test_index, test_user_ids, options, test_header, test_file_to_get_users, threshold_list = threshold_list)
        else:
          error, recall, precision, specificity = run_feature_selection_and_classification(method_to_run, train_data_subset, test_data, labels, train_index, test_index, test_user_ids, options, test_header, test_file_to_get_users, options[5], threshold_list = threshold_list)
          precisions[index] = precision
          recalls[index] = recall
          specificities[index] = specificity
          index = index + 1
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

def generate_train_test_error_for_different_kernels(train_file, test_file, output_folder):
    methods_to_run_list
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    test_labels = np.concatenate((train_labels, test_labels))
    train_labels = np.concatenate((train_labels, train_labels))
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, (train_size + test_size))
    train_test_index = range(train_size, (train_size + train_size))
    # Run linear SVM.
    train_error_1inear = run_feature_selection_and_classification(['univariate_fea_selection', 'linear_svm'],
                                                                  train_data, train_data, train_labels, train_index,
                                                                  train_index, train_user_ids,
                                                                  [7, 10, 'linear', 0, ''], train_header, '')[0]
    test_error_linear = run_feature_selection_and_classification(['univariate_fea_selection', 'linear_svm'],
                                                                 train_data, test_data, test_labels, train_index,
                                                                 test_index, test_user_ids, [7, 10, 'linear', 0, ''],
                                                                 test_header, '')[0]

    # Run poly SVM with degree 2.
    train_error_poly_2 = run_feature_selection_and_classification(['univariate_fea_selection', 'poly_svm'],
                                                                  train_data, train_data, train_labels, train_index,
                                                                  train_index, train_user_ids,
                                                                  [7, 10, 2, 0, ''], train_header, '')[0]
    test_error_poly_2 = run_feature_selection_and_classification(['univariate_fea_selection', 'poly_svm'],
                                                                 train_data, test_data, test_labels, train_index,
                                                                 test_index, test_user_ids, [7, 10, 2, 0, ''],
                                                                 test_header, '')[0]
    # Run poly SVM with degree 3.
    train_error_poly_3 = run_feature_selection_and_classification(['univariate_fea_selection', 'poly_svm'],
                                                                  train_data, train_data, train_labels, train_index,
                                                                  train_index, train_user_ids,
                                                                  [7, 10, 3, 0, ''], train_header, '')[0]
    test_error_poly_3 = run_feature_selection_and_classification(['univariate_fea_selection', 'poly_svm'],
                                                                 train_data, test_data, test_labels, train_index,
                                                                 test_index, test_user_ids, [7, 10, 3, 0, ''],
                                                                 test_header, '')[0]
    # Run poly SVM with degree 4.
    train_error_poly_4 = run_feature_selection_and_classification(['univariate_fea_selection', 'poly_svm'],
                                                                  train_data, train_data, train_labels, train_index,
                                                                  train_index, train_user_ids,
                                                                  [7, 10, 4, 0, ''], train_header, '')[0]
    test_error_poly_4 = run_feature_selection_and_classification(['univariate_fea_selection', 'poly_svm'],
                                                                 train_data, test_data, test_labels, train_index,
                                                                 test_index, test_user_ids, [7, 10, 4, 0, ''],
                                                                 test_header, '')[0]
    # Run RBF Kernel SVM.
    train_error_rbf_kernel = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                      train_data, train_data, train_labels, train_index,
                                                                      train_index, train_user_ids,
                                                                      [7, 10, 'rbf', 0, ''], train_header, '')[0]
    test_error_rbf_kernel = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                     train_data, test_data, test_labels, train_index,
                                                                     test_index, test_user_ids, [7, 10, 'rbf', 0, ''],
                                                                     test_header, '')[0]
    pyplot.figure()
    pyplot.ylim((-0.01, 1.1))
    pyplot.xlabel('Different Kernels')
    index = [1, 2, 3, 4, 5]
    pyplot.xticks(index, ['Linear SVM', 'Poly SVM Degree 2', 'Poly SVM Degree 3',
                          'Poly SVM Degree 4', 'RBF SVM'], rotation = 'vertical')
    pyplot.plot(index, [train_error_1inear, train_error_poly_2, train_error_poly_3,
                        train_error_poly_4, train_error_rbf_kernel], color = 'r')
    pyplot.plot(index, [test_error_1inear, test_error_poly_2, test_error_poly_3,
                        test_error_poly_4, test_error_rbf_kernel], color = 'b')
    pyplot.legend(['Training Error', 'Testing Error'])
    pyplot.savefig(path.join(output_folder, 'SVM_with_different_kernels.png'))            
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

