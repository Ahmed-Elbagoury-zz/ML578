import os.path as path
import os
from kfold_cross_validation import kfold_cross_validation
import numpy as np
import matplotlib.pyplot as plt

def run(c_val, training_file_path, num_feas, k):
	# classification_alg_list = ['linear_svm', 'kernel_svm']
	classification_alg_list = ['one_class_svm']
	fea_selection_alg_list = ['univariate_fea_selection', 'linear_SVC']
	kernel = 'rbf'
	for i in range(len(classification_alg_list)):
		for j in range(len(fea_selection_alg_list)):
			classification_alg = classification_alg_list[i]
			fea_selection_alg = fea_selection_alg_list[j]
			print '%s, %s' % (classification_alg, fea_selection_alg)
			output_folder = '%s/%s/%s_%s' %(classification_alg, kernel,classification_alg, fea_selection_alg)
			if not os.path.exists(output_folder):
					os.makedirs(output_folder)
			output_file = open('%s/stats.txt' %(output_folder), 'a')
			labels = ['Avg Error', 'Var Error', 'Avg Recall', 'Var recall', 'Avg Precision', 'Var Precision', 'Avg Specificity', 'Var Specificity']
			output_file.write('C\t')
			for label in labels:
				output_file.write('%s\t' %(label))
			output_file.write('\n')
			# output_file.write('C\tAvg Error\tVar Error\tAvg Recall\tVar recall\tAvg Precision\tVar Precision\tAvg Specificity\tVar Specificity\n')
			vals_to_plot = []
			for c in c_val:
				print '\tC = %f' %(c)
				mean, var = kfold_cross_validation(k, training_file_path, [fea_selection_alg, classification_alg], output_folder, [num_feas, c, kernel])
				output_file = open('%s/stats.txt' %(output_folder), 'a')
				output_file.write('%f\t' %(c))
				vals_to_plot.append([])
				for k in range(mean.shape[0]):
					output_file.write('%f\t%f\t' %(mean[k], var[k]))
					vals_to_plot[-1].extend([mean[k], var[k]])
				output_file.write('\n')
				output_file.close()
			plot_lists(c_val, labels, vals_to_plot, '%s/stats.png' %(output_folder), range(len(vals_to_plot)), False)

def plot_lists(n_vals, labels, vals, output_file, columns_to_plot, set_yticks):
		max_y = 0
		for column_index in range(len(vals[0])):
			plt.plot(n_vals, list(np.array(vals)[:, column_index]), label=labels[column_index])
		plt.xlabel('C')
		plt.legend([labels[i] for i in columns_to_plot], loc="center right")
		plt.xlim(0, n_vals[-1])
		if set_yticks:
			plt.yticks(list(np.linspace(0, 5, 21)))
			plt.ylim(0, 5)
		# plt.title('Trained using %s dataset, with batch size = 10, %d nodes'%(dataset_name, num_nodes)) 
		plt.savefig(output_file)	
		plt.clf()
if __name__ == '__main__':
	# c_val = [0.1, 1, 10, 100, 1000]
	c_val = [1, 10, 25, 40, 55, 70, 85]
	training_file_path = path.join('train_subsets', '0_train.csv')
	run(c_val, training_file_path, num_feas = 5, k = 10)
