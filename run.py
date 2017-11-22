from preprocess_transcation_table import preprocess_transcation_table
from preprocess_user_logs_table import preprocess_user_logs_table
from join_tables import join_tables
from split_data_random import split_data_random
from generate_histograms import generate_histograms
from kfold_cross_validation import kfold_cross_validation
from select_c_for_SVM_using_kfold_CV import select_c_for_SVM_using_kfold_CV
from classify_test_users import classify_test_users
from classify_test_users import test_different_number_of_samples
from classify_test_users import test_different_thresholds
from select_parameters_for_MLP_using_kfold_CV import select_parameters_for_MLP_using_kfold_CV
from select_feature_selection_Naive_Bayes_using_kfold_CV import select_feature_selection_Naive_Bayes_using_kfold_CV
from classify_test_users import test_different_subsets
from classify_test_users import generate_train_test_error_for_different_kernels
from classify_test_users import generate_train_test_error_for_different_MLP
import os.path as path
import os
def run (step_number):
    if step_number == 1:
        # Step 1: Preprocess transcation data.
        transcations_input_filename = 'transactions.csv'
        transcations_output_filename = 'transactions_preprocessed.csv'
        preprocess_transcation_table(transcations_input_filename,
                                     transcations_output_filename)
    elif step_number == 2:
        # Step 2: Preprocess user_logs data
        user_logs_input_filename = 'user_logs.csv'
        user_logs_output_filename = 'user_logs_preprocessed.csv'
        preprocess_user_logs_table(user_logs_input_filename,
                                   user_logs_output_filename)
    elif step_number == 3:
        # Step 3: Join all tables based on user_id
        transcations_output_filename = 'transactions_preprocessed.csv'
        user_logs_output_filename = 'user_logs_preprocessed.csv'
        train_input_filename = 'train.csv'
        member_input_filename = 'members.csv'
        joined_table_output_filename = 'joined_train_data.csv'
        join_tables(transcations_output_filename, user_logs_output_filename,
                    train_input_filename, member_input_filename,
                    joined_table_output_filename)
    elif step_number == 4:
        # Step 4: Split joined table file into 10 subsets randomly.
        split_number = 10
        train_data_joined = 'joined_train_data.csv'
        output_folder = 'train_subsets'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        train_ratio = 0.8
        split_data_random(train_data_joined, split_number, output_folder,
                          train_ratio)
    elif step_number == 5:
        # Step 5: Generate features histogram.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'histograms'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        k = 10
        kfold_cross_validation(k, train_file, ['features_histogram'], output_folder, [])
    elif step_number == 6:
        # Step 6: Run PCA.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'pca'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        k = 10
        kfold_cross_validation(k, train_file, ['pca'], output_folder, [])
    elif step_number == 7:
        # Step 7: Run univariate_fea_selection fearure selection.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'univariate_fea_selection'
        k = 10
        C = 10
        number_of_features_to_select = 7
        stats_vals = kfold_cross_validation(k, train_file, ['univariate_fea_selection'],
                                            output_folder, [number_of_features_to_select, C])
    elif step_number == 8:
        # Step 8: Run linear_SVC fearure selection.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'linear_SVC'
        k = 10
        sparsity_param = 0.002
        C = 10
        stats_vals = kfold_cross_validation(k, train_file, ['linear_SVC'],
                                            output_folder, [sparsity_param, C])
    elif step_number == 9:
        # Step 9: Run 10 fold cross validation to choose C for linear SVM with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'linear_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 'linear'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 10:
        # Step 10: Run 10 fold cross validation to choose C for linear SVM with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'linear_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 'linear'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)
    elif step_number == 11:
        # Step 11: Run 10 fold cross validation to choose C for kernel SVM with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'kernel_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 'rbf'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 12:
        # Step 12: Run 10 fold cross validation to choose C for kernel SVM with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'kernel_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 'rbf'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)
    elif step_number == 13:
        # Step 13: Run 10 fold cross validation to choose C for one class linear SVM with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'one_class_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 'linear'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 14:
        # Step 14: Run 10 fold cross validation to choose C for one class linear SVM with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'one_class_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 'linear'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)
    elif step_number == 15:
        # Step 15: Run 10 fold cross validation to choose C for one class kernel SVM with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'one_class_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 'rbf'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 16:
        # Step 16: Run 10 fold cross validation to choose C for one class kernel SVM with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'one_class_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 'rbf'
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)
    elif step_number == 17:
        # Step 17: For submitting on Kaggle, join the Kaggle test data with the other tables.
        transcations_output_filename = 'transactions_preprocessed.csv'
        user_logs_output_filename = 'user_logs_preprocessed.csv'
        train_input_filename = 'sample_submission.csv'
        member_input_filename = 'members.csv'
        joined_table_output_filename = 'joined_test_data.csv'
        join_tables(transcations_output_filename, user_logs_output_filename,
                    train_input_filename, member_input_filename,
                    joined_table_output_filename)
    elif step_number == 18:
        # Step 18: Classify Kaggle test user, using SVM with C learned form 10 fold cross validation.
        train_file = 'joined_train_data.csv'
        test_file = 'joined_test_data.csv'
        test_file_to_get_users = 'sample_submission.csv'
        methods_to_run = ['univariate_fea_selection', 'preceptron']
        number_of_features_to_select = 14
        C = 10
        kernel = 'linear'
        write_prediction = 1
        prediction_file = 'prediction_naive_MLP_all_10_200.csv'
        options = [number_of_features_to_select, C, kernel, write_prediction, prediction_file]
        classify_test_users(train_file, test_file, methods_to_run, test_file_to_get_users, options)
    elif step_number == 19:
        # Step 19: Run 10 fold cross validation to choose C for linear SVM with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'linear_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 'linear'
        class_1_weight = 10
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select, class_1_weight)
    elif step_number == 20:
        # Step 20: Run 10 fold cross validation to choose C for linear SVM with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'linear_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 'linear'
        class_1_weight = 10
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param, class_1_weight)
    elif step_number == 21:
        # Step 21: Run 10 fold cross validation to choose C for kernel SVM with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'kernel_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 'rbf'
        class_1_weight = 10
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select, class_1_weight)
    elif step_number == 22:
        # Step 22: Run 10 fold cross validation to choose C for kernel SVM with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'kernel_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 'rbf'
        class_1_weight = 10
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param, class_1_weight)
    elif step_number == 23:
        # Step 23: 10 fold CV for MLP using univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        params_list = [(10,10,10), (10,10,10,10,10), (10,10,10,10,10,10,10,10,10,10),
                       (50,50,50), (50,50,50,50,50), (50,50,50,50,50,50,50,50,50,50),
                       (100,100,100), (100,100,100,100,100), (100,100,100,100,100,100,100,100,100,100)]
        classification_alg = 'preceptron'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = ''
        select_parameters_for_MLP_using_kfold_CV(train_file, kernel, classification_alg,
                                                 fea_selection_alg, params_list, k,
                                                 number_of_features_to_select)
    elif step_number == 24:
        # Step 24: 10 fold CV for MLP using linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        params_list = [(10,10,10), (10,10,10,10,10), (10,10,10,10,10,10,10,10,10,10),
                       (50,50,50), (50,50,50,50,50), (50,50,50,50,50,50,50,50,50,50),
                       (100,100,100), (100,100,100,100,100), (100,100,100,100,100,100,100,100,100,100)]
        classification_alg = 'preceptron'
        fea_selection_alg = 'linear_SVC'
        kernel = ''
        select_parameters_for_MLP_using_kfold_CV(train_file, kernel, classification_alg,
                                                 fea_selection_alg, params_list, k,
                                                 sparsity_param)
    elif step_number == 25:
        # Step 25: 10 fold CV for Naive Bayes using univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        classification_alg = 'naive_bayes'
        fea_selection_alg = 'univariate_fea_selection'
        select_feature_selection_Naive_Bayes_using_kfold_CV(train_file, classification_alg, fea_selection_alg,
                                                            k, number_of_features_to_select)
    elif step_number == 26:
        # Step 26: 10 fold CV for Naive Bayes using linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        classification_alg = 'naive_bayes'
        fea_selection_alg = 'linear_SVC'
        select_feature_selection_Naive_Bayes_using_kfold_CV(train_file, classification_alg, fea_selection_alg,
                                                            k, sparsity_param)
    
    elif step_number == 27:
        # Step 27: Test different number of samples for different classification algorithms.
        # Parameters and feature selection methods are selected based on CV best in specificty.
        train_file = path.join('train_subsets', '0_train.csv')
        test_file = path.join('train_subsets', '0_test.csv')
        test_file_to_get_users = 'sample_submission.csv'
        methods_to_run_list = [['linear_SVC', 'linear_svm'], ['univariate_fea_selection', 'kernel_svm'],
                               ['linear_SVC', 'one_class_svm'], ['linear_SVC', 'one_class_svm'],
                               ['linear_SVC', 'linear_svm'],['univariate_fea_selection', 'kernel_svm'],
                               ['univariate_fea_selection', 'naive_bayes'], ['linear_SVC', 'preceptron']]
        write_prediction = 0
        prediction_file = ''
        number_of_features_to_select = 7
        sparsity_param = 0.002
        linear_SVM_options = [sparsity_param, 55, 'linear', write_prediction, prediction_file]
        kernel_SVM_options = [number_of_features_to_select, 70, 'rbf', write_prediction, prediction_file]
        one_class_linear_SVM_options = [sparsity_param, 1, 'linear', write_prediction, prediction_file]
        one_class_kernel_SVM_options = [sparsity_param, 85, 'rbf', write_prediction, prediction_file]
        class_weight_linear_SVM_options = [sparsity_param, 85, 'linear', write_prediction, prediction_file, 10]
        class_weight_kernel_SVM_options = [number_of_features_to_select, 1, 'rbf', write_prediction, prediction_file, 10]
        naive_bayes_options = [number_of_features_to_select, 0, '', write_prediction, prediction_file]
        preceptron_options = [sparsity_param, (50, 50, 50, 50, 50, 50, 50, 50, 50, 50), '', write_prediction, prediction_file]
        options_list = [linear_SVM_options, kernel_SVM_options, one_class_linear_SVM_options, one_class_kernel_SVM_options,
                        class_weight_linear_SVM_options, class_weight_kernel_SVM_options, naive_bayes_options,
                        preceptron_options]
        different_number_of_samples = range(2000, 22000, 2000)
        output_folder = 'different_number_of_samples_experiment'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        test_different_number_of_samples(train_file, test_file, methods_to_run_list, test_file_to_get_users, options_list,
                                     different_number_of_samples, output_folder)
    elif step_number == 28:
        # Step 28: Test different training subsets for different classification algorithms.
        # Parameters and feature selection methods are selected based on CV best in specificty.
        train_folder = 'train_subsets'
        test_file = path.join('train_subsets', '0_test.csv')
        test_file_to_get_users = 'sample_submission.csv'
        methods_to_run_list = [['linear_SVC', 'linear_svm'], ['univariate_fea_selection', 'kernel_svm'],
                               ['linear_SVC', 'one_class_svm'], ['linear_SVC', 'one_class_svm'],
                               ['linear_SVC', 'linear_svm'],['univariate_fea_selection', 'kernel_svm'],
                               ['univariate_fea_selection', 'naive_bayes'], ['linear_SVC', 'preceptron']]
        write_prediction = 0
        prediction_file = ''
        number_of_features_to_select = 7
        sparsity_param = 0.002
        linear_SVM_options = [sparsity_param, 55, 'linear', write_prediction, prediction_file]
        kernel_SVM_options = [number_of_features_to_select, 70, 'rbf', write_prediction, prediction_file]
        one_class_linear_SVM_options = [sparsity_param, 1, 'linear', write_prediction, prediction_file]
        one_class_kernel_SVM_options = [sparsity_param, 85, 'rbf', write_prediction, prediction_file]
        class_weight_linear_SVM_options = [sparsity_param, 85, 'linear', write_prediction, prediction_file, 10]
        class_weight_kernel_SVM_options = [number_of_features_to_select, 1, 'rbf', write_prediction, prediction_file, 10]
        naive_bayes_options = [number_of_features_to_select, 0, '', write_prediction, prediction_file]
        preceptron_options = [sparsity_param, (50, 50, 50, 50, 50, 50, 50, 50, 50, 50), '', write_prediction, prediction_file]
        options_list = [linear_SVM_options, kernel_SVM_options, one_class_linear_SVM_options, one_class_kernel_SVM_options,
                        class_weight_linear_SVM_options, class_weight_kernel_SVM_options, naive_bayes_options,
                        preceptron_options]
        different_number_of_samples = range(2000, 22000, 2000)
        output_folder = 'different_subsets_experiment'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        train_files_count = 10
        test_different_subsets(train_folder, test_file, methods_to_run_list, test_file_to_get_users,
                               options_list, output_folder, train_files_count)
    elif step_number == 29:
        # Step 29: Test different thresholds and plot the ROC curve for different classification algorithms.
        # Parameters and feature selection methods are selected based on CV best in specificty.
        threshold_list = [0.25, 0.5, 0.75, 1]
        train_file = path.join('train_subsets', '0_train.csv')
        test_file = path.join('train_subsets', '0_test.csv')
        test_file_to_get_users = 'sample_submission.csv'
        methods_to_run_list = [['linear_SVC', 'linear_svm'], ['univariate_fea_selection', 'kernel_svm'],
                               ['linear_SVC', 'one_class_svm'], ['linear_SVC', 'one_class_svm'],
                               ['linear_SVC', 'linear_svm'],['univariate_fea_selection', 'kernel_svm'],
                               ['univariate_fea_selection', 'naive_bayes'], ['linear_SVC', 'preceptron']]
        write_prediction = 0
        prediction_file = ''
        number_of_features_to_select = 7
        sparsity_param = 0.002
        linear_SVM_options = [sparsity_param, 55, 'linear', write_prediction, prediction_file]
        kernel_SVM_options = [number_of_features_to_select, 70, 'rbf', write_prediction, prediction_file]
        one_class_linear_SVM_options = [sparsity_param, 1, 'linear', write_prediction, prediction_file]
        one_class_kernel_SVM_options = [sparsity_param, 85, 'rbf', write_prediction, prediction_file]
        class_weight_linear_SVM_options = [sparsity_param, 85, 'linear', write_prediction, prediction_file, 10]
        class_weight_kernel_SVM_options = [number_of_features_to_select, 1, 'rbf', write_prediction, prediction_file, 10]
        naive_bayes_options = [number_of_features_to_select, 0, '', write_prediction, prediction_file]
        preceptron_options = [sparsity_param, (50, 50, 50, 50, 50, 50, 50, 50, 50, 50), '', write_prediction, prediction_file]
        options_list = [linear_SVM_options, kernel_SVM_options, one_class_linear_SVM_options, one_class_kernel_SVM_options,
                        class_weight_linear_SVM_options, class_weight_kernel_SVM_options, naive_bayes_options,
                        preceptron_options]
        different_number_of_samples = range(2000, 22000, 2000)
        output_folder = 'different_thresholds_experiment'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        test_different_thresholds(train_file, test_file, methods_to_run_list, test_file_to_get_users, options_list, threshold_list, output_folder, class_1_weight = 1)
    elif step_number == 30:
        # Step 30: Run 10 fold cross validation to choose C for poly SVM degree 2 with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 2
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 31:
        # Step 31: Run 10 fold cross validation to choose C for poly SVM degree 2 with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 2
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)
    elif step_number == 32:
        # Step 32: Run 10 fold cross validation to choose C for poly SVM degree 3 with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 3
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 33:
        # Step 33: Run 10 fold cross validation to choose C for poly SVM degree 3 with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 3
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)
    elif step_number == 34:
        # Step 34: Run 10 fold cross validation to choose C for poly SVM degree 4 with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 4
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 35:
        # Step 35: Run 10 fold cross validation to choose C for poly SVM degree 4 with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 4
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)

    elif step_number == 36:
        # Step 36: Run 10 fold cross validation to choose C for poly SVM degree 4 with univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = 10
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, number_of_features_to_select)
    elif step_number == 37:
        # Step 37: Run 10 fold cross validation to choose C for poly SVM degree 4 with linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        c_vals = [1, 10, 25, 40, 55, 70, 85]
        classification_alg = 'poly_svm'
        fea_selection_alg = 'linear_SVC'
        kernel = 10
        select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg, fea_selection_alg,
                                        c_vals, k, sparsity_param)
    elif step_number == 38:
        # Step 38: Report training and testing errors for different SVM kernels.
        train_file = path.join('train_subsets', '0_train.csv')
        test_file = path.join('train_subsets', '0_test.csv')
        output_folder = 'different_SVM_kernels'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        generate_train_test_error_for_different_kernels(train_file, test_file, output_folder)

    elif step_number == 39:
        # Step 39: 10 fold CV for MLP using univariate_fea_selection.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        number_of_features_to_select = 7
        params_list = [tuple([50]*3), tuple([50]*5), tuple([50]*10),
                       tuple([50]*13), tuple([50]*15), tuple([50]*20)]
        classification_alg = 'preceptron'
        fea_selection_alg = 'univariate_fea_selection'
        kernel = ''
        select_parameters_for_MLP_using_kfold_CV(train_file, kernel, classification_alg,
                                                 fea_selection_alg, params_list, k,
                                                 number_of_features_to_select)
    elif step_number == 40:
        # Step 40: 10 fold CV for MLP using linear_SVC.
        train_file = path.join('train_subsets', '0_train.csv')
        k = 10
        sparsity_param = 0.002
        params_list = [tuple([50]*3), tuple([50]*5), tuple([50]*10),
                       tuple([50]*13), tuple([50]*15), tuple([50]*20)]
        classification_alg = 'preceptron'
        fea_selection_alg = 'linear_SVC'
        kernel = ''
        select_parameters_for_MLP_using_kfold_CV(train_file, kernel, classification_alg,
                                                 fea_selection_alg, params_list, k,
                                                 sparsity_param)
    elif step_number == 41:
        # Step 41: Report training and testing errors for different MLP layers.
        train_file = path.join('train_subsets', '0_train.csv')
        test_file = path.join('train_subsets', '0_test.csv')
        output_folder = 'different_MLP_kernels'
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        generate_train_test_error_for_different_MLP(train_file, test_file, output_folder)
 
