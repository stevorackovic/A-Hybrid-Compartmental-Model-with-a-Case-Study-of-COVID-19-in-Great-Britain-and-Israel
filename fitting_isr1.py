# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 21:17:05 2021

@author: BIGMATH

In this script we perform 5 simulations with optimal parameters estimated for
GBR, but applied to ISR data. The idea is to see how usefull is parameter transfer.

"""

import numpy as np
import pandas as pd
from Model2 import run_model2
from help_functions import get_p_e_vec, moving_average, data_handling, plot_mean_std, plot_simulations
import os    
os.chdir(r'C:\Users\Stevo\Desktop\ECMI_competition\CODE')
Data_isr = pd.read_csv('data_ISR.csv', index_col = 0)
total_pop,flag_wp_sch, flag_hard_lock, vaccines, tests,testing_policy,contact_tracing,true_new_cases,true_new_deaths, true_hosp = data_handling(Data_isr)
flag_wp_sch = flag_wp_sch.values
tracing = contact_tracing
policy = testing_policy

###############################################################################

M, reg_sz = 10, 50000
reg_sizes = [reg_sz for i in range(M)]
kids_perc = .27

true_deaths_ma    = moving_average(true_new_deaths,7)
true_new_cases_ma = moving_average(true_new_cases,7)
true_hosp_ma      = moving_average(true_hosp,7)

sim_population = M*reg_sz
pop_factor = total_pop/sim_population
test_num = (tests*sim_population/total_pop).astype(int)
num_vac = (vaccines*sim_population/total_pop).astype(int)

###############################################################################
repetition_number = 5  # in order to study variation of parameters
num_days = 465         # to run full simulation

v_eff = .9   # vaccine eff known
p_i   = 1/13 # probability that an exposed node becomes infected. This should be reciprocal to the lenght of incubation period (in days).
p_r   = 1/10 # probability that an infected node recovers (reciprocal to the average lenght of a recovery).
p_s   = 1/14 # probability of a susceptible or exposed node leaving the quarantine. This value should be reciprocal to the lenght of quarantine (in days)

p_sy_h = 1/25
p_h_d  = 1/50
p_h_r  = 1/10
p_sy_t = 0.5
p_asy  = .7
p_sy_d = 1/350

w = [1,5,2]
p_e_vec =  get_p_e_vec(Data_isr, w, p_e_max=.18, p_e_min=.08)
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

os.chdir(r'C:\Users\Stevo\Desktop\ECMI_competition\Predictions')
np.save('isr_dH_mtx.npy',  dH_mtx)
np.save('isr_dE_mtx.npy',  dE_mtx)
np.save('isr_dR_mtx.npy',  dR_mtx)
np.save('isr_dD_mtx.npy',  dD_mtx)
np.save('isr_dAsy_mtx.npy',dAsy_mtx)
np.save('isr_dSy_mtx.npy', dSy_mtx)
np.save('isr_cH.npy',      cH)
np.save('isr_dPos.npy',    dPos)

