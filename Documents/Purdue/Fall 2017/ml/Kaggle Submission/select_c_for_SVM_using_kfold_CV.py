import os.path as path
import os
from kfold_cross_validation import kfold_cross_validation
from plot_cross_validation_SVM import plot_cross_validation_SVM
import random
import numpy as np
def select_c_for_SVM_using_kfold_CV(train_file, kernel, classification_alg,
                                    fea_selection_alg, c_vals, k,
                                    number_of_features_to_select):
        random.seed(0)
        output_folder = '%s/%s/%s_%s' %(classification_alg, kernel, classification_alg, fea_selection_alg)
        if not path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = open('%s/stats.txt' %(output_folder), 'a')
        labels = ['Avg Error', 'Var Error', 'Avg Recall', 'Var recall', 'Avg Precision', 'Var Precision', 'Avg Specificity', 'Var Specificity']
        output_file.write('C\t')
        for label in labels:
            output_file.write('%s\t' %(label))
        output_file.write('\n')
        mean_vals = []
        error_vals = []
        for c in c_vals:
            print '\tC = %f' %(c)
            stats_results = kfold_cross_validation(k, train_file, [fea_selection_alg, classification_alg], output_folder,
                                                   [number_of_features_to_select, c, kernel, 0])
            output_file = open('%s/stats.txt' %(output_folder), 'a')
            output_file.write('%f\t' %(c))
            mean_vals.append([])
            error_vals.append([])
            for stat_record in stats_results:
                mean_vals[-1].append(stat_record[0])
                error_vals[-1].append(stat_record[1])
                output_file.write('%f\t%f\t' %(stat_record[0], stat_record[1]))
            output_file.write('\n')
            output_file.close()
        plot_cross_validation_SVM(k, c_vals, labels, np.array(mean_vals), np.array(error_vals), '%s/stats' %(output_folder), range(len(mean_vals)), False)
