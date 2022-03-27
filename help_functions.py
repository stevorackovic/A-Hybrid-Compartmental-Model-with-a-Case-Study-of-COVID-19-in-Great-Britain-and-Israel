# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 11:57:43 2021

@author: BIGMATH
"""

import matplotlib.pyplot as plt
import numpy as np

def get_p_e_vec(data, w, p_e_max=.4, p_e_min=.05):
    restrictions = ['stay_home_restrictions','school_closing','workplace_closing']
    w_restrictions = np.average(data[restrictions].values,axis=1,weights=w)
    d = 20
    w_r_delay = np.append(np.zeros(d),(w_restrictions))[:-d]
    p_e_f_d = (w_r_delay)*(p_e_min-p_e_max)/(3-0)+p_e_max
    window = 10
    p_e_f_smooth =  moving_average(p_e_f_d,window)
    p_e_f_smooth_compl = np.append(p_e_f_smooth,p_e_f_d[-window+1:])
    return p_e_f_smooth_compl

def moving_average(arr,window_width):
    cumsum_vec = np.cumsum(np.insert(arr, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def data_handling(data):    
    df = data.copy()   
    #school work flag
    df.loc[df['school_closing']==2, 'school_closing'] = 1
    df.loc[df['workplace_closing']==2, 'workplace_closing'] = 1
    sch_close = df['school_closing']
    wp_close = df['workplace_closing']    
    flag_wp_sch= wp_close.astype(str) + sch_close.astype(str)
    # hard lockdown flag
    stay_home = df['stay_home_restrictions']
    flag_hard_lock = stay_home>1
    flag_hard_lock = flag_hard_lock.values        
    vaccines = df.vaccines_new.values
    tests = df.tests_new.values
    testing_policy = df.testing_policy.values
    contact_tracing = df.contact_tracing.values
    total_pop = df['population'].iloc[0]       
    # ground truth
    df_gt = df[['confirmed_new','deaths_new','hosp']]
    df_gt = df_gt.mask(df_gt.lt(0)).ffill().fillna(0)
    true_new_cases = df_gt['confirmed_new'].values  
    true_new_deaths = df_gt['deaths_new'].values  
    true_hosp = df_gt['hosp'].values          
    return total_pop,flag_wp_sch, flag_hard_lock, vaccines, tests,testing_policy,contact_tracing,true_new_cases,true_new_deaths, true_hosp
   
def plot_mean_std(Matrix,pop_factor,ground_truth,title):
    vector_mean = moving_average(Matrix.mean(axis=0)*pop_factor,7)
    vector_std = moving_average(Matrix.std(axis=0)*pop_factor,7)
    plt.figure(figsize=(10,5))
    plt.plot(vector_mean,c='b', label = 'Predicted')
    plt.plot(vector_mean+vector_std, c='b', alpha=.3)
    plt.plot(vector_mean-vector_std, c='b', alpha=.3)
    plt.plot(ground_truth,c='r', label = 'Observed')
    plt.title(title)
    plt.legend()       
    plt.show()
    
def plot_simulations(Matrix,pop_factor,ground_truth,title,repetition_number):
    plt.figure(figsize=(10,5))
    for i in range(repetition_number):
        vector = moving_average(Matrix[i]*pop_factor,7)
        if i ==0:
            plt.plot(vector,c='b', label = 'Predicted')
        else:
            plt.plot(vector,c='b')
    plt.plot(ground_truth,c='r', label = 'Observed')
    plt.title(title)
    plt.legend()       
    plt.show()
    
def plot_the_three(Matrix1,Matrix2,Matrix3,pop_factor,ground_truth,title):
    vector_mean1 = moving_average(Matrix1.mean(axis=0)*pop_factor,7)
    vector_mean2 = moving_average(Matrix2.mean(axis=0)*pop_factor,7)
    vector_mean3 = moving_average(Matrix3.mean(axis=0)*pop_factor,7)
    plt.figure(figsize=(10,5))
    plt.plot(vector_mean1,c='sienna', label = 'V_eff=0.9')
    plt.plot(vector_mean2,c='seagreen', label = 'V_eff=0.75')
    plt.plot(vector_mean3,c='cornflowerblue', label = 'V_eff=0')
    plt.title(title)
    plt.legend()       
    plt.show()
