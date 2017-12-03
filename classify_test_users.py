import numpy as np
import csv
from kfold_cross_validation import run_feature_selection_and_classification
from matplotlib import pyplot
from os import path
import os
import matplotlib.pyplot as plt

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
                                             options, test_header, test_file_to_get_users, class_1_weight = class_1_weight)
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
                                                                                                 class_1_weight = options[5])
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
                                                                                                     class_1_weight = options[5])
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

def plot_roc(fea_selection, classification, file_name, specificity, sensitivity):
    specificity_index = sorted(enumerate(specificity), key = lambda tup : tup[1])
    specificity = [tup[1] for tup in specificity_index]
    sensitivity = [sensitivity[tup[0]] for tup in specificity_index]
    auc = np.trapz(sensitivity, x=specificity)
    plt.clf()
    plt.xlim((0, 1))
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('%s, %s. AUC  = %f' %(fea_selection, classification, auc))
    plt.plot(specificity,sensitivity)
    plt.savefig(file_name)

def generate_ROC(train_file, test_file, threshold_list = -1):
    roc_prefix = 'ROC'
    if not os.path.exists(roc_prefix):
        os.makedirs(roc_prefix)
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    train_test_labels = np.concatenate((train_labels, test_labels))
    train_train_labels = np.concatenate((train_labels, train_labels))
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, (train_size + test_size))
    train_test_index = range(train_size, (train_size + train_size))
    
    test_error,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, tuple([50] * 10), '', 0],
                                                                 test_header, '', threshold_list = threshold_list)
    fea_selection_method = 'L$_1$-based fea selection'
    classification_method = 'Perceptron'
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)
    print '>>linear_SVC, preceptron'
    print '\t', 'test_error', test_error
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity

    c = 1
    test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'one_class_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, c, 'linear', 0, ''],
                                                                 test_header, '', threshold_list = threshold_list)
    fea_selection_method = 'L$_1$-based fea selection'
    classification_method = 'One-Class Linear SVM'
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)
    print '>>linear_SVC, one class linear_svm, %d' %(c)
    print '\t', 'test_error_linear', test_error_linear
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity

    c = 85
    test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'one_class_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, c, 'rbf', 0, ''],
                                                                 test_header, '', threshold_list = threshold_list)
    fea_selection_method = 'L$_1$-based fea selection'
    classification_method = 'One-Class Kernel SVM (RBF)'
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)

    print '>>linear_SVC, one class rbf_svm, %d' %(c)
    print '\t', 'test_error_linear', test_error_linear
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity

#-----------------------
    class_1_weight = 1
    c = 55
    test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'linear_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, c, 'linear', 0, ''],
                                                                 test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
    fea_selection_method = 'L$_1$-based fea selection'
    classification_method = 'Linear SVM'
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)
    print '>>linear_SVC, linear_svm, %d. class_1_weight = %d' %(c, class_1_weight)
    print '\t', 'test_error_linear', test_error_linear
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity


    c = 70
    test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [7, c, 'rbf', 0, ''],
                                                                 test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
    fea_selection_method = 'Chi-squared fea selection'
    classification_method = 'Kernel SVM (RBF)'
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)

    print '>>univariate_fea_selection, linear_svm, %d. class_1_weight = %d' %(c, class_1_weight)
    print '\t', 'test_error_linear', test_error_linear
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity

    c = 85
    class_1_weight = 10
    test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'linear_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, c, 'linear', 0, ''],
                                                                 test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)

    fea_selection_method = 'L$_1$-based fea selection'
    classification_method = 'Linear SVM, class 1 weight = %d' %(class_1_weight)
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s_%d_class_1_weight.png' %(roc_prefix, fea_selection_method, classification_method, class_1_weight), specificity, recall)

    print '>>linear_SVC, linear_svm, %d. class_1_weight = %d' %(c, class_1_weight)
    print '\t', 'test_error_linear', test_error_linear
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity


    c = 1
    test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [7, c, 'rbf', 0, ''],
                                                                 test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
    
    fea_selection_method = 'Chi-squared fea selection'
    classification_method = 'Kernel SVM (RBF), class 1 weight = %d' %(class_1_weight)
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s_%d_class_1_weight.png' %(roc_prefix, fea_selection_method, classification_method, class_1_weight), specificity, recall)

    print '>>univariate_fea_selection, linear_svm, %d. class_1_weight = %d' %(c, class_1_weight)
    print '\t', 'test_error_linear', test_error_linear
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity


def generate_test_error_for_kernels(train_file, test_file, threshold_list = -1):
    roc_prefix = 'ROC'
    if not os.path.exists(roc_prefix):
        os.makedirs(roc_prefix)
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    train_test_labels = np.concatenate((train_labels, test_labels))
    train_train_labels = np.concatenate((train_labels, train_labels))
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, (train_size + test_size))
    train_test_index = range(train_size, (train_size + train_size))
    
    test_error,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, tuple([50] * 10), '', 0],
                                                                 test_header, '', threshold_list = threshold_list)
    fea_selection_method = 'L$_1$-based fea selection'
    classification_method = 'Perceptron'
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)
    print '>>linear_SVC, preceptron'
    print '\t', 'test_error', test_error
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity

    test_error,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids,  [7, tuple([50] * 5), '', 0],
                                                                 test_header, '', threshold_list = threshold_list)
    fea_selection_method = 'Chi-squared fea selection'
    classification_method = 'Perceptron'
    plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)
    print 'univariate_fea_selection, preceptron'
    print '\t', 'test_error', test_error
    print '\t', 'recall', recall
    print '\t', 'precision', precision
    print '\t', 'specificity', specificity
    
    c_list = [(55, 55), (1, 1)]
    for c in c_list:
      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'one_class_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [7, c[0], 'linear', 0, ''],
                                                                   test_header, '', threshold_list = threshold_list)
      fea_selection_method = 'Chi-squared fea selection'
      classification_method = 'Linear SVM'
      plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)
      print '>>univariate_fea_selection, one class linear_svm, %d' %(c[0])
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'one_class_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [0.002, c[1], 'linear', 0, ''],
                                                                   test_header, '', threshold_list = threshold_list)
      fea_selection_method = 'L$_1$-based fea selection'
      classification_method = 'One-Class Linear SVM'
      plot_roc(fea_selection_method, classification_method, '%s/%s_%s.png' %(roc_prefix, fea_selection_method, classification_method), specificity, recall)
      print '>>linear_SVC, one class linear_svm, %d' %(c[1])
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

    c_list = [(70, 70), (85, 85)]
    for c in c_list:
      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'one_class_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [7, c[0], 'rbf', 0, ''],
                                                                   test_header, '', threshold_list = threshold_list)
      print '>>univariate_fea_selection, one class rbf_svm, %d' %(c[0])
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'one_class_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [0.002, c[1], 'rbf', 0, ''],
                                                                   test_header, '', threshold_list = threshold_list)
      print '>>linear_SVC, one class rbf_svm, %d' %(c[1])
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

#-----------------------

    c_list = [(25, 85), (10, 55)]
    class_1_weight = 1
    for c in c_list:
      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'linear_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [7, c[0], 'linear', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>univariate_fea_selection, linear_svm, %d. class_1_weight = %d' %(c[0], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'linear_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [0.002, c[1], 'linear', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>linear_SVC, linear_svm, %d. class_1_weight = %d' %(c[1], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity


    c_list = [(1, 1), (70, 55)]
    for c in c_list:
      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [7, c[0], 'rbf', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>univariate_fea_selection, linear_svm, %d. class_1_weight = %d' %(c[0], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'kernel_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [0.002, c[1], 'rbf', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>linear_SVC, linear_svm, %d. class_1_weight = %d' %(c[1], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

    c_list = [(10, 25), (10, 85)]
    class_1_weight = 10
    for c in c_list:
      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'linear_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [7, c[0], 'linear', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>univariate_fea_selection, linear_svm, %d. class_1_weight = %d' %(c[0], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'linear_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [0.002, c[1], 'linear', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>linear_SVC, linear_svm, %d. class_1_weight = %d' %(c[1], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity


    c_list = [(40, 55), (1, 1)]
    for c in c_list:
      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [7, c[0], 'rbf', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>univariate_fea_selection, linear_svm, %d. class_1_weight = %d' %(c[0], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity

      test_error_linear,recall, precision, specificity = run_feature_selection_and_classification(['linear_SVC', 'kernel_svm'],
                                                                   train_data, test_data, train_test_labels, train_index,
                                                                   test_index, test_user_ids, [0.002, c[1], 'rbf', 0, ''],
                                                                   test_header, '', class_1_weight = class_1_weight, threshold_list = threshold_list)
      print '>>linear_SVC, linear_svm, %d. class_1_weight = %d' %(c[1], class_1_weight)
      print '\t', 'test_error_linear', test_error_linear
      print '\t', 'recall', recall
      print '\t', 'precision', precision
      print '\t', 'specificity', specificity


def generate_train_test_error_for_different_kernels(train_file, test_file, output_folder):
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    train_test_labels = np.concatenate((train_labels, test_labels))
    train_train_labels = np.concatenate((train_labels, train_labels))
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, (train_size + test_size))
    train_test_index = range(train_size, (train_size + train_size))
    # Run linear SVM.
    train_error_1inear = run_feature_selection_and_classification(['linear_SVC', 'linear_svm'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                  [0.002, 55, 'linear', 0, ''], train_header, '')[0]
    test_error_linear = run_feature_selection_and_classification(['linear_SVC', 'linear_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, 55, 'linear', 0, ''],
                                                                 test_header, '')[0]

    # Run poly SVM with degree 2.
    train_error_poly_2 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                  [0.002, 40, 2, 0, ''], train_header, '')[0]
    test_error_poly_2 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, 40, 2, 0, ''],
                                                                 test_header, '')[0]
    # Run poly SVM with degree 3.
    train_error_poly_3 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                  [0.002, 70, 3, 0, ''], train_header, '')[0]
    test_error_poly_3 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, 70, 3, 0, ''],
                                                                 test_header, '')[0]
    # Run poly SVM with degree 4.
    train_error_poly_4 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                  [0.002, 55, 4, 0, ''], train_header, '')[0]
    test_error_poly_4 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, 55, 4, 0, ''],
                                                                 test_header, '')[0]

    # Run poly SVM with degree 10.
    train_error_poly_10 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                  [0.002, 1, 10, 0, ''], train_header, '')[0]
    test_error_poly_10 = run_feature_selection_and_classification(['linear_SVC', 'poly_svm'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, 1, 10, 0, ''],
                                                                 test_header, '')[0]
    
    # Run RBF Kernel SVM.
    train_error_rbf_kernel = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                      train_data, train_data, train_train_labels, train_index,
                                                                      train_test_index, train_user_ids,
                                                                      [7, 70, 'rbf', 0, ''], train_header, '')[0]
    test_error_rbf_kernel = run_feature_selection_and_classification(['univariate_fea_selection', 'kernel_svm'],
                                                                     train_data, test_data, train_test_labels, train_index,
                                                                     test_index, test_user_ids, [7, 70, 'rbf', 0, ''],
                                                                     test_header, '')[0]
    pyplot.figure()
    pyplot.xlabel('Different Kernels')
    index = [1, 2, 3, 4, 5, 6]
    pyplot.xticks(index, ['Linear', 'Poly Degree 2', 'Poly Degree 3',
                          'Poly Degree 4', 'Poly Degree 10', 'RBF'], fontsize=7)
    pyplot.plot(index, [train_error_1inear, train_error_poly_2, train_error_poly_3,
                        train_error_poly_4, train_error_poly_10, train_error_rbf_kernel], color = 'r')
    pyplot.plot(index, [test_error_linear, test_error_poly_2, test_error_poly_3,
                        test_error_poly_4, test_error_poly_10, test_error_rbf_kernel], color = 'b')
    pyplot.legend(['Training Error', 'Testing Error'])
    pyplot.savefig(path.join(output_folder, 'SVM_with_different_kernels.png'))            
    pyplot.close()

def generate_train_test_error_for_different_MLP(train_file, test_file, output_folder):
    train_data, train_header, train_labels, train_user_ids = get_data(train_file)
    test_data, test_header, test_labels, test_user_ids = get_data(test_file)
    train_test_labels = np.concatenate((train_labels, test_labels))
    train_train_labels = np.concatenate((train_labels, train_labels))
    train_size = len(train_labels)
    test_size = len(test_labels)
    train_index = range(train_size)
    test_index = range(train_size, (train_size + test_size))
    train_test_index = range(train_size, (train_size + train_size))
    # Run MLP with 3 layer.
    train_error_3 = run_feature_selection_and_classification(['linear_SVC', 'preceptron'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                 [0.002, tuple([50] * 3), '', 0], train_header, '')[0]
    test_error_3 = run_feature_selection_and_classification(['linear_SVC', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids, [0.002, tuple([50] * 3), '', 0],
                                                                 test_header, '')[0]

    # Run MLP with 5 layer.
    train_error_5 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                  [7, tuple([50] * 5), '', 0], train_header, '')[0]
    test_error_5 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids,  [7, tuple([50] * 5), '', 0],
                                                                 test_header, '')[0]
    # Run MLP with 10 layer.
    train_error_10 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                   [7, tuple([50] * 10), '', 0], train_header, '')[0]
    test_error_10 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids,  [7, tuple([50] * 10), '', 0],
                                                                 test_header, '')[0]
    # Run MLP with 20 layer.
    train_error_13 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                   [7, tuple([50] * 13), '', 0], train_header, '')[0]
    test_error_13 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids,  [7, tuple([50] * 13), '', 0],
                                                                 test_header, '')[0]

    # Run MLP with 50 layer.
    train_error_15 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                  train_data, train_data, train_train_labels, train_index,
                                                                  train_test_index, train_user_ids,
                                                                   [7, tuple([50] * 15), '', 0], train_header, '')[0]
    test_error_15 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                 train_data, test_data, train_test_labels, train_index,
                                                                 test_index, test_user_ids,  [7, tuple([50] * 15), '', 0],
                                                                 test_header, '')[0]
    
    # Run MLP with 100 layer.
    train_error_20 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                      train_data, train_data, train_train_labels, train_index,
                                                                      train_test_index, train_user_ids,
                                                                       [7, tuple([50] * 20), '', 0], train_header, '')[0]
    test_error_20 = run_feature_selection_and_classification(['univariate_fea_selection', 'preceptron'],
                                                                     train_data, test_data, train_test_labels, train_index,
                                                                     test_index, test_user_ids,  [7, tuple([50] * 20), '', 0],
                                                                     test_header, '')[0]
    pyplot.figure()
    pyplot.xlabel('Different Number of Layers')
    index = [1, 2, 3, 4, 5, 6]
    pyplot.xticks(index, ['3', '5', '10', '13', '15', '20'], fontsize=7)
    pyplot.plot(index, [train_error_3, train_error_5, train_error_10,
                        train_error_13, train_error_15, train_error_20], color = 'r')
    pyplot.plot(index, [test_error_3, test_error_5, test_error_10,
                        test_error_13, test_error_15, test_error_20], color = 'b')
    pyplot.legend(['Training Error', 'Testing Error'])
    pyplot.savefig(path.join(output_folder, 'MLP_with_different_layers.png'))            
    pyplot.close()

        
  

if __name__ == '__main__':
    # train_file = 'train_subsets/0_train.csv'
    # test_file = 'test_subsets/0_test.csv'
    
    # number_of_features_to_select = 7
    
    # # methods_to_run = ['linear_SVC', 'kernel_svm']
    # # C = 40
    # # class_1_weight = 10
    # # kernel = 'rbf'

    # # write_prediction = 0
    # # prediction_file = ''
    # # sparsity_param = 0.002
    # # options = [sparsity_param, C, kernel, write_prediction, prediction_file]

    # write_prediction = 0
    # prediction_file = ''
    # methods_to_run = ['linear_SVC', 'preceptron']
    # options = [number_of_features_to_select, [100] * 10, '', write_prediction, prediction_file]
    # classify_test_users(train_file, test_file, methods_to_run, '', options)

    train_file = path.join('train_subsets', '0_train.csv')
    test_file = path.join('train_subsets', '0_test.csv')
    generate_test_error_for_kernels(train_file, test_file)

