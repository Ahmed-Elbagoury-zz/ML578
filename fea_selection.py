from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from helper import load_data

def univariate_fea_selection(X, y, k):
	return SelectKBest(chi2, k=k).fit_transform(X, y)

def my_SelectFromModel(X, y, sparsity_param):
	lsvc = LinearSVC(C=sparsity_param, penalty="l1", dual=False).fit(X, y)
	model = SelectFromModel(lsvc, prefit=True)
	X_new = model.transform(X)
	return X_new

if __name__ == '__main__':
	input_path = '0_train.csv'
	k = 10
	data, labels = load_data(input_path)
	print len(data), len(labels)
	# low_dim_data = univariate_fea_selection(data, labels, k)
	low_dim_data = my_SelectFromModel(data, labels, 0.01)
	print low_dim_data.shape