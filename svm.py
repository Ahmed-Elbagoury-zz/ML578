import csv
import numpy as np
from sklearn import svm

def load_data(input_path):
	rows = []
	with open(input_path, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			rows.append(row)
	labels = [int(row[1]) for row in rows[1:]]
	rows = [[float(num) for num in row[2:]] for row in rows[1:]]
	return np.array(rows), np.array(labels)

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