import numpy as np
import scipy.stats
# import flexcode
import math
import random
# from flexcode.regression_models import NN # for univariate response data Y
import matplotlib.pyplot as plt
# from sklearn.datasets import make_s_curve
import torch
from scipy.spatial.distance import pdist, squareform

class PCP_VCR(object):
    def __init__(self, device="cpu", alpha=0.1, n_sample_K=40,y_dim=1):
        self.device = device
        self.alpha = alpha
        self.n_sample_K = n_sample_K
        self.y_dim = y_dim
    def rank(self, Y,Y_hat, k_neighbor = 5):
        # Y shape: (n_sample, 1, y_dim)
        # Y hat shape: (n_sample, n_sample_K, y_dim)
        n = Y.shape[0]
        y_dim = self.y_dim
        #############  ranking of Y_hat 
        Y_hat_ranked = []
        for idx in range(n):
            pairwise_matrix = squareform(pdist(Y_hat[idx]))
            knn_avg_dist = np.mean(np.sort(pairwise_matrix, axis=1)[:, :k_neighbor + 1], axis=1)
            rank = np.argsort(knn_avg_dist)
            Y_hat_ranked.append(Y_hat[idx][rank])
        Y_hat_ranked = np.array(Y_hat_ranked)
        return Y_hat_ranked  # shape: (n_sample, n_sample_K, y_dim)
    def compute_dist_matrix(self,Y,Y_hat):
        # Y shape: (n_sample, 1, y_dim)
        # Y hat shape: (n_sample, n_sample_K, y_dim)
        
        dist_matrix = np.linalg.norm(Y - Y_hat,axis=2)
        return dist_matrix

    def calibrate(self, dist_matrix_cal, num_iter=100, position = 0):
        '''
        Compute the (1 - alpha) threshold `E_q` for non-conformity score vectors using an approximation algorithm.
        Parameters:
        -----------
        dist_matrix_cal : ndarray of shape (n_cal, n_sample_K)
        The ranked distance matrix for the calibration data.
        num_iter : int, default=100
            The number of iterations for the approximation algorithm.
        position : int, default=0
            The index of the initialized position in `E_q` to adjust.
    
        Returns:
        --------
        E_q_free : ndarray of shape (n_sample_K,)
            The optimized threshold `E_q` that satisfies the (1 - alpha) coverage.
        '''
        alpha = self.alpha; n_sample = self. n_sample_K
        min_radius = float('inf')

        # Step 1: Initialize E_q values
        E_q_init = np.min(dist_matrix_cal,axis=0) # or np.zeros(n_sample)
        E_q_free = E_q_init.copy()
        
        #Step 2: Adjust `position` entry to satisfy coverage
        ### i.e., increase beta_position till just achieve target coverage
        for val in np.sort(dist_matrix_cal[:, position])[::-1]:
            E_q_init[position]=val
            c = np.mean(np.max(dist_matrix_cal - E_q_init <= 0,axis=1))
            if c>=1-alpha:
                r = np.sum(E_q_init ** self.y_dim)
                if r <=min_radius:
                    min_radius=r.copy()
                    E_q_free = E_q_init.copy()
                    coverage = c
            else:
                break

        # Step 3: Iterative refinement of `E_q`, decrease `position` entry of E_q to and increase a random entries of E_q to satisfy coverage.
        l = list(range(0,n_sample))
        del l[position]
    
        update_count = 0
        E_q_init = E_q_free.copy()
        step_back=1
        for iteration in range(num_iter):
            update = False
            
         # Shuffle valid positions for randomized updates
            l_rand = np.random.permutation(l)
            for rand_pos in l_rand:
                E_q_init = E_q_free.copy()
                sorted_value = np.sort(dist_matrix_cal[:, position])
                idx = (sorted_value==E_q_init[position]).nonzero()[0]
                ### make sure there is still room to decrease E_q_init[position], if idx ==0, there is no room for decrease.
                if idx>=step_back:
                    ### increase beta_position (decrease quantile on ``position''-entry), with a step_back size.
                    E_q_init[position] = sorted_value[idx-step_back]
                else:
                    break
                    
                # Only consider values greater than or equal to the current entry for efficienct computation.
                for val in np.sort(dist_matrix_cal[dist_matrix_cal[:,rand_pos]>=E_q_init[rand_pos],rand_pos]):
                    E_q_init[rand_pos] = val
                    c = np.mean(np.max(dist_matrix_cal - E_q_init <= 0,axis=1))
                    if c>=1-alpha:
                        r = np.sum(E_q_init ** self.y_dim)
                        if r <=min_radius:
                            update = True
                            min_radius=r.copy()
                            E_q_free = E_q_init.copy() 
                            coverage = c
                        break

                if update: update_count+=1
                    
            # Adjust step_back for subsequent iterations
            if not update:
                step_back+=1
            else: 
                if step_back!=1:
                    step_back-=1
        # print('number of update: ', update_count)
        return E_q_free
        
    def empirical_coverage(self, dist_matrix,radius):
        coverage = np.mean(np.max(dist_matrix<=radius,axis=1))
        return coverage
    
    def pcp_radius(self,dist_matrix): 
        '''
        dist_matrix: distance matrix for calibration data
        alpha: target error rate (1-alpha is the target coverage rate)
        '''
        E_q = np.quantile(np.min(dist_matrix,axis=1),q=1-self.alpha)
        return E_q
    def pcp_coverage(self,dist_matrix, radius):
        '''
        dist_matrix: distance matrix for testing data 
        radius: radius calculated for PCP from "pcp_radius" function
        '''
        coverage = np.mean(np.max(dist_matrix - radius <= 0,axis=1))
        return coverage


def get_overlap_length(intervals):
    temp_tuple = intervals
    temp_tuple.sort(key=lambda interval: interval[0])
    merged = [temp_tuple[0]]
    for current in temp_tuple:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    l=0
    for x in merged:
        l+= x[1]-x[0]
    return l 
 
def get_intervals(radius,i,y):
    
    if isinstance(radius, float):
        radius = [radius]*n_sample
    R = []
    for j in range(n_sample):
        l = (y[i,j]-radius[j])
        u = (y[i,j]+radius[j])
        R.append([l,u])
    return R


def get_coverage_length_overlap(radius,Y_test):
    """
    Y_test in the shape of (n_test,n_sample)
    """
    
    n_test = Y_test.shape[0];n_sample = Y_test.shape[1]
    if isinstance(radius, float):
        radius = [radius]*n_sample
    
    Length=[]
    for i in range(n_test):
        I = []
        for j in range(n_sample):
            l = (Y_test[i,j]-radius[j])
            u = (Y_test[i,j]+radius[j])
            I.append([l,u])
        coverage_length=get_overlap_length(I)
        Length.append(coverage_length) 
    return Length


def get_coverage_area_overlap(radius, Y_test,Y_data,dimension_y=2):
    """
    Y_test in the shape of (n_test,n_sample)
    """
    n_test = Y_test.shape[0]     
    random_data = np.random.uniform(low=[min(Y_data[:,i]) for i in range(dimension_y)], high=[max(Y_data[:,i]) for i in range(dimension_y)], size=(100**dimension_y,dimension_y))
    if isinstance(radius, float):
        pass
    else:radius=radius.numpy()
    efficenciy=[]    
    for i in range(n_test):
        coverage_region = np.mean(np.any(np.linalg.norm(random_data.reshape(100**dimension_y,dimension_y,1)-Y_test[i],axis=1)<=radius,axis=1))
        efficenciy.append(coverage_region)
    return efficenciy
