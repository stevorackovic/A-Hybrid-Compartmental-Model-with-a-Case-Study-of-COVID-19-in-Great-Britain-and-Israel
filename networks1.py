# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:55:23 2021

@author: BIGMATH
"""
import networkx as nx
import numpy as np

def create_networks2(size,kids_perc,k=4):
    population = size

    L_hard_lock = nx.watts_strogatz_graph(n = population, k = k, p = 0)
    L_s = nx.watts_strogatz_graph(n = population, k = k, p = 0.013)
    L_m = nx.watts_strogatz_graph(n = population, k = k, p = 0.026)
    L_l = nx.watts_strogatz_graph(n = population, k = k, p = 0.054)

    R_s = nx.fast_gnp_random_graph(n = population, p = 3.5/(population-1))
    R_m = nx.fast_gnp_random_graph(n = population, p = 4/(population-1))
    R_l = nx.fast_gnp_random_graph(n = population, p = 5/(population-1))

    kids_all_schools = list(np.random.choice(L_hard_lock.nodes(), int(size*kids_perc)))

    return L_hard_lock, L_s, L_m, L_l, R_s ,R_m, R_l, kids_all_schools