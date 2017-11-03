from sklearn.decomposition import PCA
import numpy as np

class MyPCA(object):
	def fit(self, data, n_components):
		self.pca = PCA(n_components=n_components)
		self.pca.fit(data)

	def transform(self, data):
		return self.pca.transform(data)



if __name__ == '__main__':
	myPca = MyPCA()
	X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	myPca.fit(X, n_components= 2)
	y = np.array([[1, 2], [3, 4], [5, 6]])
	print myPca.transform(y)


# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# iris = load_iris()
# X, y = iris.data, iris.target
# print X.shape
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SelectFromModel
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X)
# print X_new.shape