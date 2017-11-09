from preprocess_transcation_table import preprocess_transcation_table
from preprocess_user_logs_table import preprocess_user_logs_table
from join_tables import join_tables
from split_data_random import split_data_random
from generate_histograms import generate_histograms
from kfold_cross_validation import kfold_cross_validation
from select_c_for_SVM_using_kfold_CV import select_c_for_SVM_using_kfold_CV
from classify_test_users import classify_test_users
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
        methods_to_run = ['univariate_fea_selection', 'linear_svm']
        number_of_features_to_select = 7
        C = 1000
        kernel = 'linear'
        write_prediction = 1
        prediction_file = 'prediction.csv'
        options = [number_of_features_to_select, C, kernel, write_prediction, prediction_file]
        classify_test_users(train_file, test_file, methods_to_run, test_file_to_get_users, options)
