import numpy as np
from sklearn import svm
from helper import sigmoid

def train_kernel_svm (data, labels, c, class_1_weight = 1):
    clf = svm.SVC(C = c, class_weight = {1: class_1_weight}, random_state=0)
    clf.fit(data, labels.ravel())
    return clf

def train_poly_kernel_svm (data, degree, labels, c, class_1_weight = 1):
    print('in poly svm, training data = '+str(data.shape))
    clf = svm.SVC(C = c, kernel = 'poly', degree = degree, random_state=0, verbose = 2)
    clf.fit(data, labels.ravel())
    return clf

def train_linear_svm (data, labels, c, class_1_weight = 1):
    lin_clf = svm.LinearSVC(C = c, class_weight = {1: class_1_weight}, random_state=0)
    lin_clf.fit(data, labels.ravel())
    return lin_clf

def train_one_class_svm(data, kernel):
    clf = svm.OneClassSVM(kernel=kernel, random_state=0)
    clf.fit(data)
    return clf
def classify_one_class_svm(clf, test_data, threshold = -1):
    if threshold == -1:
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
    else:
        pred_confidence = clf.decision_function(test_data)
        pred = [1 if (val if threshold == 0 else sigmoid(val)) > threshold else 0 for val in pred_confidence]
        return pred, pred_confidence
def classify (clf, test_data, threshold = 0):
    pred_confidence = clf.decision_function(test_data)
    pred = [clf.classes_[1] if (val if threshold == 0 else sigmoid(val)) > threshold else clf.classes_[0] for val in pred_confidence]
    return pred, pred_confidence
