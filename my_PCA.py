from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from helper import load_data
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

if __name__ == '__main__':
	input_path = '0_train.csv'
	data, labels = load_data(input_path)
	print data.shape
	myKernelPCA = MyKernelPCA()
	myKernelPCA.fit(data, n_components = 2)