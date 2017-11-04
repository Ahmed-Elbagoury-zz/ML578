import numpy as np
from sklearn import svm
def train_kernel_svm (data, labels, c):
	clf = svm.SVC(C = c)
	clf.fit(data, labels.ravel())
	return clf

def train_linear_svm (data, labels, c):
	lin_clf = svm.LinearSVC(C = c)
	lin_clf.fit(data, labels.ravel())
	return lin_clf

def classify (clf, test_data):
	pred = clf.decision_function(test_data)
	pred = [clf.classes_[1] if val > 0 else clf.classes_[0] for val in pred]
	return pred