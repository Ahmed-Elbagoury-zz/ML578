import numpy as np
def calculate_performance_measures(predicted_labels, actual_labels):
    # Positive class is class 0, which means user will renew.
    # Negative class is class 1, which means user won't renew.
    predicted_labels = np.array(predicted_labels)
    actual_labels = np.array([cur_list[0] for cur_list in list(actual_labels)])
    TP = np.sum(np.logical_and(predicted_labels == 0, actual_labels == 0))
    FP = np.sum(np.logical_and(predicted_labels == 0, actual_labels == 1))
    FN = np.sum(np.logical_and(predicted_labels == 1, actual_labels == 0))
    TN = np.sum(np.logical_and(predicted_labels == 1, actual_labels == 1))
    if (TP + FP + FN + TN) != 0:
	error = float(FP + FN) / float(TP + FP + FN + TN)
    else:
	error = 0.0
    if (TP + FN) != 0:
	recall = TP / float(TP + FN)
    else:
	recall = 0.0
    if (TP + FP) != 0:
	precision = TP / float(TP + FP)
    else:
	precision = 0.0
    if (TN + FP) != 0:
	specificity = TN / float(TN + FP)
    else:
	specificity = 0
    #print 'TP = ', TP, 'FP = ', FP, 'FN = ', FN, 'TN = ', TN
    #print '%f/%f+%f = %f' %(TP, TP, FP, TP / float(TP + FP))
    return error, recall, precision, specificity
