import numpy as np
from sklearn import svm
def train_kernel_svm (data, labels, c):
	clf = svm.SVC(C = c, random_state=0)
	clf.fit(data, labels.ravel())
	return clf

def train_linear_svm (data, labels, c):
	lin_clf = svm.LinearSVC(C = c, random_state=0)
	lin_clf.fit(data, labels.ravel())
	return lin_clf

def train_one_class_svm(data, kernel):
        clf = svm.OneClassSVM(kernel=kernel, random_state=0)
	clf.fit(data)
	return clf
def classify_one_class_svm(clf, test_data):
	pred_confidence = clf.predict(test_data)
	"""
	leave label is 1
	stay label 0
	"""
	"""
        predicted value
        -1 will leave
	1 will stay
	"""
	pred = [1 if val == 1 else 0 for val in pred_confidence]
	return pred, pred_confidence
def classify (clf, test_data):
	pred_confidence = clf.decision_function(test_data)
	pred = [clf.classes_[1] if val > 0 else clf.classes_[0] for val in pred_confidence]
	return pred, pred_confidence
