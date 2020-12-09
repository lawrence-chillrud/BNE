"""
Run ensemble model where the GPs have been approximated by RFF on Boston data set.
"""
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.linalg import cholesky, cho_solve

import pickle as pk

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import matplotlib.pyplot as plt
import seaborn as sns

import rff_time

sys.path.extend(['/Users/adityamakkar/Desktop/research/Ensemble/calibre-master'])

#from calibre.model import gaussian_process as gp
#from calibre.model import tailfree_process as tail_free
import calibre.util.utils_patch as tail_free
#from calibre.model import gp_regression_monotone as gpr_mono
#from calibre.model import adaptive_ensemble

#from calibre.inference import mcmc

#from calibre.calibration import score

#import calibre.util.misc as misc_util
#import calibre.util.metric as metric_util
import calibre.util.visual as visual_util
#import calibre.util.matrix as matrix_util
#import calibre.util.ensemble as ensemble_util
#import calibre.util.calibration as calib_util

#import calibre.util.experiment_pred as pred_util

#from calibre.util.inference import make_value_setter

import jax.numpy as jnp
from jax import grad, jit, vmap, random, hessian
import jax
#JAX_ENABLE_X64=True

DATA_DIR = "data_ca"
RESULTS_DIR = "./results/calibre_2d_annual_pm25_example"

MODEL_DICTIONARY = {"root": ["AV", "GS", "CM"]}

DEFAULT_LOG_LS_WEIGHT = np.log(0.35).astype(np.float32)
DEFAULT_LOG_LS_RESID = np.log(0.1).astype(np.float32)

####################
### Prepare data ###
####################

print('Preparing data...')

# Read the training data file
training_data = pd.read_csv("{}/CA_clean_training_set_2010_2016_v2.csv".format(DATA_DIR))

# (508,3) numpy array containing the longitude, latitude and time as features.
X_train = np.asarray(training_data[["lon", "lat", "time"]].values.tolist()).astype(np.float32)
print('X_train.shape = ', X_train.shape)
# (508,) numpy array containing observed pm25 values.
y_train = np.asarray(training_data["pm25_obs"].tolist()).astype(np.float32)
assert y_train.shape[0] == X_train.shape[0]

# Dictionary mapping model name to its training features in a (43,2) array.
# It's the same as X_train for all models.
base_train_feat = dict()
# Dictionary mapping model name to its training predictions in a (43,) array.
base_train_pred = dict()
for model_name in tail_free.get_leaf_model_names(MODEL_DICTIONARY):
    base_train_feat[model_name] = X_train
    base_train_pred[model_name] = training_data["pred_{}".format(model_name)].values

""" 1. Prepare prediction data dictionary """
# Dictionary mapping model name to its validation features in a (5060384, 3) array.
# The features array is the same for all models.
base_valid_feat = dict()
# Dictionary mapping model name to its validation predictions in a (5060384,) array
base_valid_pred = dict()
for model_name in tail_free.get_leaf_model_names(MODEL_DICTIONARY):
	# A DataFrame with 5060384 rows and 4 columns. 
	# col1 is row number. col2 is latitude. col3 is longitude. col4 is pm25
    data_pd = pd.read_csv("{}/CA_{}_2010_2016_align.csv".format(DATA_DIR, model_name))
    base_valid_feat[model_name] = np.asarray(data_pd[["lon", "lat", "time"]].values.tolist()).astype(np.float32)
    base_valid_pred[model_name] = np.asarray(data_pd["pm25"].tolist()).astype(np.float32)

X_valid = base_valid_feat[model_name] # model_name = 'AV' but this matrix would be same for any.
N_pred = X_valid.shape[0] # = 5060384
print('N_pred = ', N_pred)

""" 2. Save the preprocessed data. """
os.makedirs(os.path.join(RESULTS_DIR, 'base'), exist_ok=True)

with open(os.path.join(RESULTS_DIR, 'base/base_train_feat.pkl'), 'wb') as file:
    pk.dump(base_train_feat, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(RESULTS_DIR, 'base/base_train_pred.pkl'), 'wb') as file:
    pk.dump(base_train_pred, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(RESULTS_DIR, 'base/base_valid_feat.pkl'), 'wb') as file:
    pk.dump(base_valid_feat, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(RESULTS_DIR, 'base/base_valid_pred.pkl'), 'wb') as file:
    pk.dump(base_valid_pred, file, protocol=pk.HIGHEST_PROTOCOL)

""" 3. Standardize data """
# standardize
X_centr = np.mean(X_valid, axis=0)
X_scale = np.max(X_valid, axis=0) - np.min(X_valid, axis=0)

X_valid = (X_valid - X_centr) / X_scale
X_train = (X_train - X_centr) / X_scale

print('Data prepared.')

#############################################
### Fit GP using RFF on the training data ###
#############################################

# Hyperparameters
D = 100
lmbda = 0.5
sigma_n = 0.1
length_scale_xy = np.log(0.052)
length_scale_t = np.log(0.0625)

# RFF model to get the features.
rff_model = rff_time.RFF(X=X_train, D=D, sigma=[length_scale_xy, length_scale_t])
# N x D matrix
Z = rff_model.get_Z(X_train)

@jit
def predictions(w, Z, base_train_pred, lmbda, sigma_n):
	"""
	# Input arguments
	w : (K+1) x D matrix storing the weights to be optimized.
	Z : N x D matrix storing the data.

	# Returns
	y_hat : Predictions array.
	"""
	# N x (K+1 matrix)
	# w_k[i, k] = inner product of Z[i] and w[k]
	w_k = jnp.matmul(Z, w.T)
	# N x K matrix such that each row sums to 1.
	s = jax.nn.softmax(w_k[:, :-1])
	# N x K matrix where each column of s has been multiplied with the corresponding base model prediction.
	s_f = jnp.vstack((s[:, 0] * base_train_pred['AV'], s[:, 1] * base_train_pred['GS'], s[:, 2] * base_train_pred['CM'])).T
	# N x 1 matrix containing the predictions.
	y_hat = jnp.sum(s_f, axis=1) + w_k[:, -1]
	return y_hat

@jit
def loss(w, Z, y_train, base_train_pred, lmbda, sigma_n):
	"""
	# Input arguments
	w : (K+1) x D matrix storing the weights to be optimized.
	Z : N x D matrix storing the data.

	# Returns
	ans : A scalar loss value.
	"""
	# The prior term
	ans = (lmbda / 2) * jnp.sum(jnp.square(jnp.linalg.norm(w, axis=1)))
	y_hat = predictions(w, Z, base_train_pred, lmbda, sigma_n)
	# Likelihood term
	ans += (1.0 / sigma_n**2) * jnp.sum(jnp.square(y_train - y_hat))
	return ans

# Initialize the weights matrix.
key = random.PRNGKey(0)
w = random.normal(key, shape=(4, D))

grad_loss = jit(grad(loss))

# Train using gradient descent.
MAX_ITERS = 10000
loss_list = []
for i in range(MAX_ITERS):
	loss_i = loss(w, Z, y_train, base_train_pred, lmbda, sigma_n)
	loss_list.append(loss_i)
	if i % 200 == 0:
		print('Iteration ', i, ': ', loss_i)
	w -= grad_loss(w, Z, y_train, base_train_pred, lmbda, sigma_n) * 0.00001

print('MAP found.')

#----------------------------------------------------------------------------------------------
print('Computing the covariance matrix...')
# Covariance matrix calculation
hessian_loss = jit(hessian(loss))
cov_matrix_inv_weird = np.array(-hessian_loss(w, Z, y_train, base_train_pred, lmbda, sigma_n))
cov_matrix_inv = np.empty(shape=(4 * D, 4 * D))
for i in range(4 * D):
	for j in range(4 * D):
		cov_matrix_inv[i, j] = cov_matrix_inv_weird[i // D, i % D, j // D, j % D]
cov_matrix = np.linalg.inv(cov_matrix_inv)
print('Covariance matrix computed.')
#----------------------------------------------------------------------------------------------

###################
### Predictions ###
###################

print('Doing predictions...')

def predictions_test(w, Z, base_valid_pred, lmbda, sigma_n, start, finish):
	"""
	# Input arguments
	w : (K+1) x D matrix storing the trained weights.
	Z : N x D matrix storing the data.

	# Returns
	y_hat : Predictions array.
	"""
	# N x (K+1 matrix)
	# w_k[i, k] = inner product of Z[i] and w[k]
	w_k = np.matmul(Z, w.T)
	# N x K matrix such that each row sums to 1.
	s = jax.nn.softmax(w_k[:, :-1])
	# N x K matrix where each column of s has been multiplied with the corresponding base model prediction.
	s_f = np.vstack((s[:, 0] * base_valid_pred['AV'][start : finish], s[:, 1] * base_valid_pred['GS'][start : finish], s[:, 2] * base_valid_pred['CM'][start : finish])).T
	# N x 1 matrix containing the predictions.
	y_hat = np.sum(s_f, axis=1) + w_k[:, -1]
	return y_hat


M = X_valid.shape[0] # 5060384
BATCH_SIZE = 5000
SPLITS = int(np.ceil(M / BATCH_SIZE))
y_map = np.array([])
for i in range(SPLITS):
	if i % 200 == 0:
		print(i)
	Z_test_i = rff_model.get_Z(X_valid[i * BATCH_SIZE : min((i+1) * BATCH_SIZE, M)])
	y_map = np.hstack((y_map, predictions_test(w, Z_test_i, base_valid_pred, lmbda, sigma_n, i*BATCH_SIZE, min((i+1) * BATCH_SIZE, M))))

#----------------------------------------------------------------------------------------------
NUM_SAMPLES = 50
w_samples = np.random.multivariate_normal(np.reshape(w, newshape=(4 * D, )), cov_matrix, size=NUM_SAMPLES)
y_samples = np.empty(shape=(NUM_SAMPLES, M))
for i in range(NUM_SAMPLES):
	if i % 5 == 0:
		print('MC sample = ', i)
	y_i = np.array([])
	for i in range(SPLITS):
		Z_test_i = rff_model.get_Z(X_valid[i * BATCH_SIZE : min((i+1) * BATCH_SIZE, M)])
		y_i = np.hstack((y_i, predictions_test(w, Z_test_i, base_valid_pred, lmbda, sigma_n, i*BATCH_SIZE, min((i+1) * BATCH_SIZE, M))))
	y_samples[i] = y_i

print('Computing emprical variance...')
y_emp_var = np.var(y_samples, axis=0)
#----------------------------------------------------------------------------------------------

print('Predictions done.')
print('Visualizing...')

#################
### Visualize ###
#################
visual_util.posterior_heatmap_2d(y_map, X=X_valid, X_monitor=X_train, cmap='RdYlGn_r', norm=None, norm_method="percentile", save_addr=None)
#----------------------------------------------------------------------------------------------
visual_util.posterior_heatmap_2d(y_emp_var, X=X_valid, X_monitor=X_train, cmap='inferno_r', norm=None, norm_method="percentile", save_addr=None)
#----------------------------------------------------------------------------------------------
