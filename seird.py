# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:24:26 2021

@author: BIGMATH
"""

import numpy as np
import networkx as nx

def seird_iteration(L,E,R,D,Asy,Sy,H,Q_Asy,Q_Sy,Q_S,Q_E,V,kids,p_e,p_i,p_r,p_sy_d,p_h_d,p_h_r,p_asy,p_sy_h,v_eff,p_s,num_vac=0,test_policy='no_testing',test_num=0,p_sy_t = .5,contact_tracing=False,p_ct=1,return_count=False):
    ''' This function performs a single iteration of a model. It returns 
    updated parameters, and optionally counts of new cases, deaths,..
    Parameters:
               L - graph representing the population
               E - list; exposed nodes of graph L
               R - list; recovered nodes of graph L
               D - list; dead nodes of graph L
             Asy - list; Asymptomatic nodes
              Sy - list; symptomatic nodes
               H - list; hospitalized nodes of graph L
           Q_Asy - list; quarantined nodes, taken from Asy
            Q_Sy - list; quarantined nodes, taken from Sy
             Q_S - list; quarantined nodes, taken from S
             Q_E - list; quarantined nodes, taken from E
               V - list; vaccinated nodes
            kids - list; we keep track of nodes that represent students, as 
                   they will not get vaccinated.
             p_e - scalar (0 < p_e < 1); a probability that a susceptible node 
                   becomes exposed (if it is a neighbour of an infected node)
             p_i - scalar (0 < p_i < 1); a probability that an exposed node 
                   becomes infected. This should be reciprocal to the lenght of 
                   incubation period (in days).
             p_r - scalar (0 < p_r < 1); a probability that an infected node 
                   recovers (reciprocal to the average lenght of a recovery).
          p_sy_d - scalar (0 < p_sy_d < 1); probability that a symptomatic node die
           p_h_d - scalar (0 < p_h_d < 1); probability that a hospitalized node die
           p_h_r - scalar (0 < p_h_d < 1); probability that a hospitalized node recovers
           p_asy - scalar (0 < p_asy < 1); probability that an infceted node 
                   gets asymptomatic
          p_sy_h - scalar (0 < p_sy_h < 1); probability that a symptomatic node 
                   gets hospitalized
           v_eff - scalar (0 < v_eff < 1); vaccine efficiency (eg. if v_eff==.9, 
                   a node is 90% less likely to get infected or symptomatic).
             p_s - scalar (0 < p_s < 1); probability of a susceptible or exposed
                   node leaving the quarantine. This value should be reciprocal
                   to the lenght of quarantine (in days)
         num_vac - int; number of vaccines given on a current day
     test_policy - string; one of three possibilities: 'no_testing', 'symptomatic' 
                   or 'random'. Default: 'no_testing'.
        test_num - number of tests performed on a current day
          p_sy_t - scalar (0 < p_sy_t < 1); a percentage of tests that is performed 
                   exclusivelly over symptomatic nodes. Deafault value: 0.5 
 contact_tracing - boolean; if True, we apply contact tracing policy, i.e. 
                   percentage of neighbors of newly positive nodes gets 
                   quaranteened. Default: False.
            p_ct - scalar (0 < p_ct < 1); a percentage of neighbors of newly 
                   positive node that gets quarantined when a contac tracing 
                   policy is applied. Default value: 1.
    return_count - boolean; if True, the function will also return counts of 
                   new exposed, infected, recovered and dead nodes.
    '''
    # We start with testing.
    # Depending on a testing policy, we have different sampling of the nodes.
    if test_policy == 'no_testing':
        positive = []
    else:
        # In case of random testings, we exclude these nodes that are already confirmed positive,
        # (which is hoospitalized, quarantined and so on).
        # Nodes that test positive get quarantined.
        if test_policy == 'random':
            # a percentage of tests is taken specifically from Sy
            positive_1 = list(set(np.random.choice(Sy,size=min(int(p_sy_t*test_num),len(Sy)),replace=False)))
            Q_Sy += positive_1
            Q_Sy = list(set(Q_Sy))
            Sy = list(set(Sy).difference(set(positive_1)))
            # and the rest is taken at random
            sample_nodes = list(set(L.nodes()).difference(H+R+D+Q_Asy+Q_Sy+Q_E+Q_S))
            tested_nodes = np.random.choice(sample_nodes,size=min(int((1-p_sy_t)*test_num),len(sample_nodes)),replace=False)
            positive_asy = list(set(tested_nodes)&set(Asy))
            positive_sy = list(set(tested_nodes)&set(Sy))
            Q_Asy += positive_asy
            Q_Asy = list(set(Q_Asy))
            Asy = list(set(Asy).difference(set(positive_asy)))
            Q_Sy += positive_sy
            Q_Sy = list(set(Q_Sy))
            Sy = list(set(Sy).difference(set(positive_sy)))
            positive = list(set(positive_asy+positive_sy))
            
        # If we test only symptomatic node, we assume that they all end up positive.
        elif test_policy == 'symptomatic':
            positive = list(set(np.random.choice(Sy,size=min(test_num,len(Sy)),replace=False)))
            Q_Sy += positive
            Q_Sy = list(set(Q_Sy))
            Sy = list(set(Sy).difference(set(positive)))            
        else:
            print('Choose correct parameter value!!!')   
            
    # Now update Quaranteened nodes - they can change their state with given probabiliteis...
    # Asymptomatic nodes can only change to recovered:
    prob_1 = np.random.rand(len(Q_Asy))
    qasy_to_R = list(np.array(Q_Asy)[np.where(prob_1<p_r)[0]])
    R += qasy_to_R
    R = list(set(R))
    Q_Asy = list(set(Q_Asy).difference(set(qasy_to_R)))
    # Symptomatic nodes can swithc to recovered, hospitalized or dead:
    prob_2 = np.random.rand(len(Q_Sy))
    qsy_to_R = list(np.array(Q_Sy)[np.where(prob_2<p_r)[0]])
    R += qsy_to_R
    R = list(set(R))
    qsy_to_D = list(np.array(Q_Sy)[np.where(prob_2>1-p_sy_d)[0]])
    D += qsy_to_D
    D = list(set(D))
    Q_Sy = list(set(Q_Sy).difference(set(qsy_to_R+qsy_to_D)))
    prob_2 = prob_2[prob_2>=p_r]
    prob_2 = prob_2[prob_2<=1-p_sy_d]
    qsy_to_H = list(np.array(Q_Sy)[np.where(prob_2<p_r+p_sy_h)[0]])
    H += qsy_to_H
    H = list(set(H))
    Q_Sy = list(set(Q_Sy).difference(set(qsy_to_H)))  
    # Exposed can get infected (i.e. go to Q_Asy or Q_Sy with respective probabilities) or leave a quaranteen
    prob_3 = np.random.rand(len(Q_E))
    qe_to_I = list(np.array(Q_E)[np.where(prob_3<p_i)[0]])
    prob_4 = np.random.rand(len(qe_to_I))
    qe_to_qasy = list(np.array(qe_to_I)[np.where(prob_4<p_asy)[0]])
    qe_to_qsy = list(np.array(qe_to_I)[np.where(prob_4>=p_asy)[0]])
    Q_E = list(set(Q_E).difference(set(qe_to_I)))  
    Q_Asy += qe_to_qasy
    Q_Asy = list(set(Q_Asy))
    Q_Sy += qe_to_qsy
    Q_Sy = list(set(Q_Sy))
    prob_3 =  prob_3[prob_3>=p_i]
    qe_to_E = list(np.array(Q_E)[np.where(prob_3>=1-p_s)[0]])
    E += qe_to_E
    E = list(set(E))
    Q_E = list(set(Q_E).difference(set(qe_to_E)))  
    # Susceptible nodes leave quaranteen with probability p_s
    qs_to_S = list(np.array(Q_S)[np.where(np.random.rand(len(Q_S))<p_s)[0]])
    Q_S = list(set(Q_S).difference(set(qs_to_S)))  
    
    
    # New Exposed nodes (S --> E):
    # probability of getting exposed additionally depends on the fact that a node was vaccinated or not.
    neighbors_all = [list(L.neighbors(node_i))[j] for node_i in Asy+Sy for j in range(len(list(L.neighbors(node_i))))]
    contact_nodes = set(neighbors_all).difference(set(R+D+Asy+Sy+H+Q_E+Q_Asy+Q_Sy+Q_S+E)) #  remove those that are not susceptible
    contact_nodes_v = list(contact_nodes & set(V))# vaccinated
    contact_nodes_nv = list(contact_nodes.difference(contact_nodes_v)) # non-vaccinated
    s_to_E = list(np.array(contact_nodes_v)[np.where(np.random.rand(len(contact_nodes_v))<p_e*(1-v_eff))[0]]) + list(np.array(contact_nodes_nv)[np.where(np.random.rand(len(contact_nodes_nv))<p_e)[0]])
    E += s_to_E
    E = list(set(E))
    
    # New infected nodes (E --> I)
    # each node that was exposed has probablity p_i to become infected
    e_to_I = list(set(np.array(E)[np.where(np.random.rand(len(E))<p_i)[0]]))
    E = list(set(E).difference(set(e_to_I)))
    e_to_Iv = list(set(e_to_I)&set(V))
    e_to_Inv = list(set(e_to_I).difference(e_to_Iv))
    prob_5_v = np.random.rand(len(e_to_Iv))
    prob_5_nv = np.random.rand(len(e_to_Inv))

    e_to_asy = list(np.array(e_to_Iv)[np.where(prob_5_v>(1-p_asy)*(1-v_eff))[0]]) + list(np.array(e_to_Inv)[np.where(prob_5_nv>(1-p_asy))[0]])
    e_to_sy = list(np.array(e_to_Iv)[np.where(prob_5_v<=(1-p_asy)*(1-v_eff))[0]]) +list(np.array(e_to_Inv)[np.where(prob_5_nv<=(1-p_asy))[0]])
    Asy += e_to_asy
    Asy = list(set(Asy))
    Sy += e_to_sy
    Sy = list(set(Sy))
    
    # Recovery, hospitalization and death (I --> R, I --> H, I --> D)
    # each infected node can recover with probability p_r... 
    #... but if it was symptomatic it can also get hospitalized (with prob. p_sy_h) or die (p_sy_d).
    # Check first for symptomatic nodes:
    prob_6 = np.random.rand(len(Sy))
    sy_to_R = list(np.array(Sy)[np.where(prob_6<p_r)[0]])
    sy_to_D = list(np.array(Sy)[np.where(prob_6>1-p_sy_d)[0]])
    R += sy_to_R
    R = list(set(R))
    D += sy_to_D
    D = list(set(D))
    Sy = list(set(Sy).difference(set(sy_to_R+sy_to_D)))
    prob_6 = prob_6[prob_6>=p_r]
    prob_6 = prob_6[prob_6<=1-p_sy_d]
    sy_to_H = list(np.array(Sy)[np.where(prob_6<p_r+p_sy_h)[0]])
    H += sy_to_H
    H = list(set(H))
    Sy = list(set(Sy).difference(set(sy_to_H)))
    # Now check for Asymptomatic:
    asy_to_R = list(np.array(Asy)[np.where(np.random.rand(len(Asy))<p_r)[0]])
    R += asy_to_R
    R = list(set(R))
    Asy = list(set(Asy).difference(set(asy_to_R)))
    
    # Some of hospitalized nodes die
    prob_7 = np.random.rand(len(H))
    h_to_D = list(np.array(H)[np.where(prob_7<p_h_d)[0]])
    D += h_to_D
    D = list(set(D))
    h_to_R = list(np.array(H)[np.where(prob_7>1-p_h_r)[0]])
    R += h_to_R
    R = list(set(R))
    H = list(set(H).difference(set(h_to_D+h_to_R)))
    
    # vaccinating (we consider only Susceptible, exposed and asymptomatic nodes):
    sample_nodes = list(set(L.nodes()).difference(H+D+Sy+Q_Asy+Q_Sy+V+kids))
    new_vaccines = np.random.choice(sample_nodes,size=min(num_vac,len(sample_nodes)),replace=False) 
    V += list(new_vaccines)
    V = list(set(V))

    # Contact tracing.
    # Check nodes that tested positive today, and set (some of) their neighbors in quaranteen    
    neighbors = [list(L.neighbors(node_i))[j] for node_i in positive for j in range(len(list(L.neighbors(node_i))))]
    neighbors = set(neighbors)
    contact_nodes = list(neighbors.difference(set(D+R+H+Q_E+Q_S+Q_Asy+Q_Sy+positive)))
    quaranteened = set(np.array(contact_nodes)[np.random.rand(len(contact_nodes))<p_ct])
    e_to_qe = list(quaranteened & set(E))
    quaranteened = quaranteened.difference(set(e_to_qe))    
    asy_to_qasy = list(quaranteened & set(Asy))
    quaranteened = quaranteened.difference(set(asy_to_qasy))
    sy_to_qsy = list(quaranteened & set(Sy))
    quaranteened = quaranteened.difference(set(sy_to_qsy))
    s_to_qs = list(quaranteened)
    Q_E += e_to_qe
    E = list(set(E).difference(set(e_to_qe)))
    Q_Asy += asy_to_qasy
    Q_Asy = list(set(Q_Asy))
    Asy = list(set(Asy).difference(set(asy_to_qasy)))
    Q_Sy += sy_to_qsy
    Q_Sy = list(set(Q_Sy))
    Sy = list(set(Sy).difference(set(sy_to_qsy)))
    Q_S += s_to_qs
    Q_S = list(set(Q_S))

            
    # Now update the counts
    new_Asy = e_to_asy
    new_Sy = e_to_sy
    new_H = qsy_to_H + sy_to_H
    new_E = s_to_E
    new_R = qasy_to_R + qsy_to_R + asy_to_R + sy_to_R + h_to_R
    new_D = qsy_to_D + sy_to_D + h_to_D
    new_H = qsy_to_H + sy_to_H
    new_Q = positive + e_to_qe + asy_to_qasy + sy_to_qsy + s_to_qs
    
    if return_count==False:
        return Asy, Sy, H, E, R, D, Q_Asy, Q_Sy, Q_S, Q_E, V
    else:
        dAsy, dSy, dE, dR, dH, dD,dQ = len(new_Asy), len(new_Sy), len(new_E), len(new_R), len(new_H), len(new_D), len(new_Q), 
        return Asy, Sy, H, E, R, D, Q_Asy, Q_Sy, Q_S, Q_E, V, dAsy, dSy, dE, dR, dH, dD, dQ, len(positive)