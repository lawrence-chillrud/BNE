import numpy as np

from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

class RFF:
	"""
	This class implements Kernel approximation using Random Fourier Features (RFF).
	Based on "Random Features for Large-Scale Kernel Machines" by Rahimi and Recht (2007).
	"""

	def __init__(self, X, D=10, sigma=[1.0, 1.0], amplitude=1.0):
		"""
		X : An N x d matrix containing N "raw" data points.
		D : Number of samples used for the approximation. This equals the dimension of the vector 
		    z in the paper.
		"""
		self.D = D
		self.sigma = sigma
		self.amplitude = amplitude
		self.X = X
		self.N , self.d = X.shape
		# This matrix contains the omega's of the paper.
		self.OMEGA = np.random.normal(size=(self.D, self.d))
		# Modifying OMEGA to account for different length scales
		self.OMEGA[:, 0:(self.d-1)] *= (1 / sigma[0])
		self.OMEGA[:, self.d - 1] *= (1 / sigma[1])
		# The b's of the paper.
		self.B = np.random.uniform(0, 2*np.pi, size=(self.D, 1))

	def get_Z(self, X):
		"""
		# Returns
		Z : An N x D matrix containing N "processed" data points.
		"""
		norm = np.sqrt((2 * self.amplitude) / self.D)
		Z = norm * np.cos(np.matmul(self.OMEGA, X.T) + self.B)
		Z = Z.T
		return Z