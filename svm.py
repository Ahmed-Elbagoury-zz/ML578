import numpy as np
from sklearn import svm
from helper import load_data
def train_svm(input_path):
	data, labels = load_data(input_path)
	clf = svm.SVC()
	clf.fit(data, labels)
	print lin_clf.decision_function(data[0])
	return clf

def train_linear_svm(input_path):
	data, labels = load_data(input_path)
	print len(data)
	lin_clf = svm.LinearSVC()
	lin_clf.fit(data, labels)
	print lin_clf.decision_function(data[0])
	return clf

if __name__ == '__main__':
	input_path = '0_train.csv'
	# train_svm(input_path)
	train_linear_svm(input_path)