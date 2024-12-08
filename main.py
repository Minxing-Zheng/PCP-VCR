import os
import time
import pickle
import numpy as np
import scipy.stats
import sys
import random
from sklearn.datasets import make_s_curve
import functions
import sklearn
import multiprocess as mp
import tqdm
import torch
from scipy.spatial.distance import pdist, squareform
import math
import flexcode
import matplotlib.pyplot as plt
import multiprocess as mp
import tqdm
import pickle
from flexcode.regression_models import NN # for univariate response data Y
import argparse

# Parse input arguments
parser = argparse.ArgumentParser(description="Run PCP VCR experiments.")
parser.add_argument('--n_sample', type=int, default=10, help="Number of samples")
parser.add_argument('--n_exp', type=int, default=10, help="Number of experiments")
parser.add_argument('--n', type=int, default=5000, help="Number of data points")
args = parser.parse_args()

n_sample = args.n_sample
n_exp = args.n_exp
n = args.n


result=[]
for exp in range(n_exp):
    x, t = make_s_curve(n_samples=n,noise=0)
    X= x[:,0].reshape(n,1);Y=x[:,2].reshape(n,1)
    train,calib,test = np.split(range(n),[int(.6*n),int(.8*n),])
    model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",regression_params={"k":50})
    # Fit and tune model
    train,validate = np.split(range(len(train)),[int(.9*len(train)),])
    model.fit(X[train], Y[train])
    model.tune(X[validate], Y[validate])
    
    
    cdes, y_grid = model.predict(X[np.concatenate((calib, test))], n_grid=100)
    normalized_probs = cdes / cdes.sum(axis=1, keepdims=True)  # Normalize probabilities once
    Y_hat = y_grid[np.array([
        np.random.choice(len(y_grid), size=n_sample, p=probs) for probs in normalized_probs
    ])]
    pcp_vcr = functions.PCP_VCR(n_sample_K = n_sample,alpha=0.1,y_dim = Y.shape[1])
    Y_cal_test = Y[np.concatenate((calib, test))].reshape(-1,1,pcp_vcr.y_dim)
    Y_hat_ranked = pcp_vcr.rank(Y_cal_test,Y_hat,k_neighbor = 4)
    dist_matrix = pcp_vcr.compute_dist_matrix(Y_cal_test,Y_hat)
    dist_matrix_rank = pcp_vcr.compute_dist_matrix(Y_cal_test,Y_hat_ranked)
    
    def get_r(pos):
        r = pcp_vcr.calibrate(dist_matrix_rank[:len(calib),:],num_iter = 300,position=pos)
        return r
        
    cpu_count = min(mp.cpu_count()-1, n_sample)
    E_q_list=[]
    for i in tqdm.tqdm(mp.Pool(cpu_count,initializer=np.random.seed).imap(get_r, range(n_sample)), total=n_sample):
        E_q_list.append(i)
    radius_list = [np.sum(x** pcp_vcr.y_dim) for x in E_q_list]
    pcp_vcr_radius = E_q_list[np.argmin(radius_list)]
    emp_coverage = pcp_vcr.empirical_coverage(dist_matrix_rank[len(calib):,:],pcp_vcr_radius)
    rank_pcp_exact_length = functions.get_coverage_length_overlap(pcp_vcr_radius,Y_hat_ranked[len(calib):])
    
    pcp_radius = pcp_vcr.pcp_radius(dist_matrix[:len(calib)])
    pcp_coverage = pcp_vcr.empirical_coverage(dist_matrix[len(calib):],pcp_radius)
    pcp_exact_length = functions.get_coverage_length_overlap(pcp_radius,Y_hat[len(calib):])
    
    result.append([emp_coverage,rank_pcp_exact_length,pcp_coverage,pcp_exact_length])
print(f"PCP-VCR empirical coverage: {np.mean([x[0] for x in result]):.3f}")
print(f"PCP-VCR empirical exact efficiency: {np.mean([x[1] for x in result]):.3f} \n")

print(f"PCP empirical coverage: {np.mean([x[2] for x in result]):.3f}")
print(f"PCP empirical exact efficiency: {np.mean([x[3] for x in result]):.3f}")
with open("result.pkl",'wb') as f:
    pickle.dump(result,f)
    
