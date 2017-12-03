import csv
import numpy as np
import math
from decimal import Decimal
import warnings
from scipy.stats import logistic

def sigmoid(x):
	# warnings.filterwarnings('error')
	# try: 
	# 	return 1.0/(1.0 + math.e**(-1 * x))
	# except Warning:
	# 	print '>>>>>>>',  x
	logistic.cdf(x)
def load_data(input_path):
	rows = []
	with open(input_path, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			rows.append(row)
	labels = [int(row[1]) for row in rows[1:]]
	rows = [[float(num) for num in row[2:]] for row in rows[1:]]
	return np.array(rows), np.array(labels)