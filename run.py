from preprocess_transcation_table import preprocess_transcation_table
from preprocess_user_logs_table import preprocess_user_logs_table
from join_tables import join_tables
from split_data_random import split_data_random
from generate_histograms import generate_histograms
from kfold_cross_validation import kfold_cross_validation
import os.path as path
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
        train_ratio = 0.8
        split_data_random(train_data_joined, split_number, output_folder,
                          train_ratio)
    elif step_number == 5:
        # Step 5: Generate features histogram.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'histograms'
        k = 10
        kfold_cross_validation(k, train_file, ['features_histogram'], output_folder, [])
    elif step_number == 6:
        # Step 6: Run PCA.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'pca'
        k = 10
        kfold_cross_validation(k, train_file, ['pca'], output_folder, [])
    elif step_number == 7:
        # Step 7: Run linear SVM with univariate_fea_selection fearure selection.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'linear_svm'
        k = 10
        number_of_features_to_select = 5
        kfold_cross_validation(k, train_file, ['univariate_fea_selection', 'linear_svm'],
                               output_folder, [number_of_features_to_select])
    elif step_number == 8:
        # Step 8: Run linear SVM with linear_SVC fearure selection.
        train_file = path.join('train_subsets', '0_train.csv')
        output_folder = 'linear_svm'
        k = 10
        sparsity_param = 0.0001
        kfold_cross_validation(k, train_file, ['linear_SVC', 'linear_svm'],
                               output_folder, [sparsity_param])
        
