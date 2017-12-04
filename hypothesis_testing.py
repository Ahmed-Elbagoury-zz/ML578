import os.path as path
import os
from kfold_cross_validation import kfold_cross_validation
from plot_cross_validation_SVM import plot_cross_validation_SVM
import random
import numpy as np
import scipy
import statsmodels.sandbox.stats.multicomp

def do_hypothesis_testing(train_file, k):
	fea_selection_list = ['linear_SVC', 'linear_SVC', 'linear_SVC', 'linear_SVC', 'univariate_fea_selection', 'linear_SVC', 'univariate_fea_selection', 'univariate_fea_selection']
	classification_list = ['preceptron', 'one_class_svm', 'one_class_svm', 'linear_svm', 'kernel_svm', 'linear_svm', 'kernel_svm', 'naive_bayes']
	options_list = [
	[0.002, tuple([50] * 10), '', 0],
	[0.002, 1, 'linear', 0, ''],
	[0.002, 85, 'rbf', 0, ''],
	[0.002, 55, 'linear', 0, ''],
	[7, 70, 'rbf', 0, ''],
	[0.002, 85, 'linear', 0, ''],
	[7, 1, 'rbf', 0, ''],
	[7, 0, '', 0, '']
	]
	class_weight_list = [1, 1, 1, 1, 1, 10, 10, 1]

	stats_results_list = []
	for i in range(len(options_list)):
		print 'i', i
		stats_results = kfold_cross_validation(k, train_file, [fea_selection_list[i], classification_list[i]], '', options_list[i], class_1_weight = class_weight_list[i], return_measures = True)
		stats_results_list += [stats_results]
	
	#doing hypothesis testing
	stats_results_list = np.array(stats_results_list)
	m, n = stats_results_list.shape
	print 'm', m, 'n', n
	x_g  = np.sum(stats_results_list) / float(n * m)
	x_bar_list = np.mean(stats_results_list, axis = 1)
	ms_within = 0
	
	for i in range(m):
		x_bar = x_bar_list[i]
		for j in range(n):
			ms_within += (stats_results_list[i][j] - x_bar) ** 2 /(float(m * (n-1)))
	df_within = m * (n-1)
	ms_between = 0
	for i in range(m):
		ms_between += n * (x_bar_list[i] - x_g)**2/ float(m-1)
	df_between = m-1
	f_stat = ms_between / ms_within	
	f = scipy.stats.f.ppf(0.95, df_between, df_within)
	# for row in stats_results_list:
	# 	for num in row:
	# 		print num,
	# 	print ''
	print 'F from table', f
	print 'f_stat', f_stat
	if f_stat > f:
		print 'Reject null hypothesis'
		"""
		In this case we have to make Post Hoc Test
		"""
		count = 0
		for i in range(m):
			for j in range(m):
				# if i == j:
				# 	continue
				diff = abs(x_bar_list[i] - x_bar_list[j])/ (float(ms_within)/n)
				diff_stat = statsmodels.sandbox.stats.multicomp.get_tukeyQcrit(n, df_within, 0.05)
				if diff > diff_stat:
					# print fea_selection_list[i], classification_list[i], options_list[i], 'class_1_weight', class_weight_list[i]
					# print 'Is statistically significant than'
					# print fea_selection_list[j], classification_list[j], options_list[j], 'class_1_weight', class_weight_list[j]
					# print '----------------'
					if x_bar_list[i] <= x_bar_list[j]:
						print 'B\t',
					else:
						print 'W\t',
				else:
					print 'Not\t',
			print ''
	else:
		print 'Accept null hypothesis'