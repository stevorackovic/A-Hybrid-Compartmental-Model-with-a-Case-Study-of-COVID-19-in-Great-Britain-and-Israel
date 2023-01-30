# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 21:17:05 2021

@author: BIGMATH

In this script, we perform five simulations with optimal parameters estimated for GBR data. 
Each simulation builds a new graph, which increases the robustness of the model. 
"""


import numpy as np
import pandas as pd
from Model import run_model
from help_functions import get_p_e_vec, moving_average, data_handling, plot_mean_std, plot_simulations
import os
data_path        = r'C:\User\...\Data'        # where to load the data from
predictions_path = r'C:\User\...\Predictions' # where to save the esimated values
os.chdir(data_path)
Data_gbr = pd.read_csv('data_GBR.csv', index_col = 0)
total_pop,flag_wp_sch, flag_hard_lock, vaccines, tests,testing_policy,contact_tracing,true_new_cases,true_new_deaths, true_hosp = data_handling(Data_gbr)
flag_wp_sch = flag_wp_sch.values
tracing = contact_tracing
policy = testing_policy

###############################################################################

M, reg_sz = 10, 50000
reg_sizes = [reg_sz for i in range(M)]
kids_perc = .18

true_deaths_ma    = moving_average(true_new_deaths,7)
true_new_cases_ma = moving_average(true_new_cases,7)
true_hosp_ma      = moving_average(true_hosp,7)

sim_population = M*reg_sz
pop_factor = total_pop/sim_population
test_num = (tests*sim_population/total_pop).astype(int)
num_vac = (vaccines*sim_population/total_pop).astype(int)

###############################################################################
repetition_number = 5  # in order to study the variation of parameters
num_days = 465         # to run a full simulation

v_eff = .9   # The vaccine efficacy (known)
p_i   = 1/13 # The probability that an exposed node becomes infected. This should be reciprocal to the length of the incubation period (in days).
p_r   = 1/10 # The probability that an infected node recovers (reciprocal to the average length of recovery).
p_s   = 1/14 # The probability of a susceptible or exposed node leaving the quarantine. This value should be reciprocal to the length of quarantine (in days)

p_sy_h = 1/25
p_h_d  = 1/50
p_h_r  = 1/10
p_sy_t = 0.5
p_asy  = .7
p_sy_d = 1/350

w = [1,5,2]
p_e_vec =  get_p_e_vec(Data_gbr, w, p_e_max=.18, p_e_min=.08)
n_I_init, n_E_init = 4, 8
day_init = 30

dH_mtx, dE_mtx, dR_mtx, dD_mtx, dAsy_mtx, dSy_mtx, cH, dPos = run_model2(M, reg_sizes, p_e_vec, p_i, p_r,
                                                                         p_sy_d, p_h_d, p_h_r, p_asy, p_sy_h, v_eff, test_num, testing_policy, num_vac, p_s, contact_tracing,
                                                                         repetition_number, num_days, flag_wp_sch, flag_hard_lock, p_sy_t, day_init,n_I_init,n_E_init,kids_perc)
   
############# Plot all simulations

plot_simulations(dD_mtx,pop_factor,true_deaths_ma,'Daily Deaths',repetition_number)
plot_simulations(dPos,pop_factor,true_new_cases_ma,'Daily Confirmed Cases',repetition_number)
plot_simulations(cH,pop_factor,true_hosp_ma,'Currently Hospitalized Cases',repetition_number)

############# Plot mean predictions with std

plot_mean_std(dD_mtx,pop_factor,true_deaths_ma,'Daily Deaths')
plot_mean_std(dPos,pop_factor,true_new_cases_ma,'Daily Confirmed Cases')
plot_mean_std(cH,pop_factor,true_hosp_ma,'Currently Hospitalized Cases')

os.chdir(predictions_path)
np.save('gbr_dH_mtx.npy',  dH_mtx)
np.save('gbr_dE_mtx.npy',  dE_mtx)
np.save('gbr_dR_mtx.npy',  dR_mtx)
np.save('gbr_dD_mtx.npy',  dD_mtx)
np.save('gbr_dAsy_mtx.npy',dAsy_mtx)
np.save('gbr_dSy_mtx.npy', dSy_mtx)
np.save('gbr_cH.npy',      cH)
np.save('gbr_dPos.npy',    dPos)

