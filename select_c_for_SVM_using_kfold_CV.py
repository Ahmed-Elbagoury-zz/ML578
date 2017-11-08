import os.path as path
import os
from kfold_cross_validation import kfold_cross_validation
from plot_cross_validation_SVM import plot_cross_validation_SVM
import random
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
        vals_to_plot = []
        for c in c_vals:
            print '\tC = %f' %(c)
            stats_vals = kfold_cross_validation(k, train_file, [fea_selection_alg, classification_alg], output_folder,
                                               [number_of_features_to_select, c, kernel, 0])
            output_file = open('%s/stats.txt' %(output_folder), 'a')
            output_file.write('%f\t' %(c))
            vals_to_plot.append([])
            for k in range(4):
                mean = stats_vals[k][0]
                var = stats_vals[k][1]
                output_file.write('%f\t%f\t' %(mean, var))
                vals_to_plot[-1].extend([mean, var])
            output_file.write('\n')
            output_file.close()
        plot_cross_validation_SVM(c_vals, labels, vals_to_plot, '%s/stats.png' %(output_folder), range(len(vals_to_plot)), False)
