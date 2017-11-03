from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

def univariate_fea_selection(X, y):
	return SelectKBest(chi2, k=2).fit_transform(X, y)

def SelectFromModel(X, y, sparsity_param):
	lsvc = LinearSVC(C=sparsity_param, penalty="l1", dual=False).fit(X, y)
	model = SelectFromModel(lsvc, prefit=True)
	X_new = model.transform(X)
	return X_new