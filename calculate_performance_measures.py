import numpy as np
def calculate_performance_measures(predicted_labels, actual_labels):
    # Positive class is class 0, which means user will renew.
    # Negative class is class 1, which means user won't renew.
    predicted_labels = np.array(predicted_labels)
    actual_labels = np.array([cur_list[0] for cur_list in list(actual_labels)])
    TP = float(np.sum(np.logical_and(predicted_labels == 0, actual_labels == 0)))
    FP = float(np.sum(np.logical_and(predicted_labels == 0, actual_labels == 1)))
    FN = float(np.sum(np.logical_and(predicted_labels == 1, actual_labels == 0)))
    TN = float(np.sum(np.logical_and(predicted_labels == 1, actual_labels == 1)))
    print 'TP:', TP, 'FP', FP, 'FN', FN, 'TN', TN
    error = (FP + FN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    return error, recall, precision, specificity
