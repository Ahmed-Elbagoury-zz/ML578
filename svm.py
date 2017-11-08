import numpy as np
from sklearn import svm
def train_kernel_svm (data, labels, c):
	clf = svm.SVC(C = c, class_weight={1: 10})
	clf.fit(data, labels.ravel())
	return clf

def train_linear_svm (data, labels, c):
	lin_clf = svm.LinearSVC(C = c, class_weight={1: 10})
	lin_clf.fit(data, labels.ravel())
	return lin_clf

def train_one_class_svm(data, kernel, nu, gamma):
	clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
	# clf = svm.OneClassSVM(nu=nu, kernel=kernel)
	clf.fit(data)
	return clf
def classify_one_class_svm(clf, test_data):
	pred = clf.predict(test_data)
	"""
	leave label is 1
	stay label 0
	"""
	pred = [1- val for val in pred]
	return pred
def classify (clf, test_data):
	pred = clf.decision_function(test_data)
	pred = [clf.classes_[1] if val > 0 else clf.classes_[0] for val in pred]
	return pred