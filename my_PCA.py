from sklearn.decomposition import PCA, KernelPCA
import numpy as np
class MyPCA(object):
	def fit(self, data, n_components):
		self.pca = PCA(n_components=n_components)
		self.pca.fit(data)

	def transform(self, data):
		return self.pca.transform(data)


class MyKernelPCA(object):
	def fit(self, data, n_components):
		#self.pca = PCA(n_components=n_components)
		self.pca = KernelPCA(n_components = n_components, kernel = 'rbf', gamma = 10)
		self.pca.fit(data)
	def transform(self, data):
		return self.pca.transform(data)