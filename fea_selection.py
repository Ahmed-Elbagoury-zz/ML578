from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

def univariate_fea_selection(X, y, k):
	selector = SelectKBest(chi2, k=k)
	selector.fit(X, y)
	indices = selector.get_support(indices=True)
	return indices

def my_SelectFromModel(X, y, sparsity_param):
	lsvc = LinearSVC(C=sparsity_param, penalty="l1", dual=False).fit(X, y.ravel())
	model = SelectFromModel(lsvc, prefit=True)
	# X_new = model.transform(X)
	index_flag = model.get_support()
	indices = [i for i in range(len(index_flag)) if index_flag[i]]
	return indices