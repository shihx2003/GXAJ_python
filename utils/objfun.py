# -*- encoding: utf-8 -*-
'''
@File    :   obj_fun.py
@Create  :   2025-04-19 13:51:11
@Author  :   shihx2003
@Version :   1.0
@Contact :   shihx2003@outlook.com
'''

# here put the import lib
import os
import re
import sys
import numpy as np
import pandas as pd

def Bias(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    bias_values = {}
    for colums in sim.columns[1:]:
        bias_value = np.sum(sim[colums].values - obs['Qobs'].values) / len(obs['Qobs'])
        bias_values[colums] = bias_value

    return bias_values

def PBias(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    pbias_values = {}
    for colums in sim.columns[1:]:

        pbias_value = 100 * (np.sum(sim[colums].values - obs['Qobs'].values) / np.sum(obs['Qobs'].values))
        pbias_values[colums] = pbias_value
    
    return pbias_values

def RMSE(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    rmse_values = {}
    for colums in sim.columns[1:]:
        rmse_value = np.sqrt(np.sum((obs['Qobs'].values - sim[colums].values)**2) / len(obs['Qobs']))
        rmse_values[colums] = rmse_value
    return rmse_values

def CC(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    cc_values = {}
    for colums in sim.columns[1:]:
        cc_value = np.corrcoef(obs['Qobs'].values, sim[colums].values)[0, 1]
        cc_values[colums] = cc_value
    return cc_values

def NSE(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    nse_values = {}
    for colums in sim.columns[1:]:
        nse_value = 1 - (np.sum((obs['Qobs'].values - sim[colums].values)**2) / np.sum((obs['Qobs'].values - np.mean(obs['Qobs'].values))**2))
        nse_values[colums] = nse_value
    return nse_values

def KGE(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    kge_values = {}
    for colums in sim.columns[1:]:

        r = np.corrcoef(obs['Qobs'].values, sim[colums].values)[0, 1]
        alpha = np.std(sim[colums].values) / np.std(obs['Qobs'].values)
        beta = np.mean(sim[colums].values) / np.mean(obs['Qobs'].values)
        
        kge_value = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
        kge_values[colums] = kge_value
    return kge_values

def MRE(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    mre_values = {}
    for colums in sim.columns[1:]:
        mre_value = ((np.max(sim[colums].values) - np.max(obs['Qobs'].values)) / np.max(obs['Qobs'].values)) * 100
        mre_values[colums] = mre_value
    return mre_values

def Tlag(obs, sim):
    obs = obs.reset_index(drop=True)
    sim = sim.reset_index(drop=True)
    tlag_values = {}
    for colums in sim.columns[1:]:

        sim_max_idx = np.argmax(sim[colums].values)
        obs_max_idx = np.argmax(obs['Qobs'].values)

        tlag_value = sim_max_idx - obs_max_idx

        tlag_values[colums] = tlag_value

    return tlag_values

def FastObjFun(obs, sim):

    bias = Bias(obs, sim)
    pbias = PBias(obs, sim)
    rmse = RMSE(obs, sim)
    cc = CC(obs, sim)
    nse = NSE(obs, sim)
    kge = KGE(obs, sim)
    mre = MRE(obs, sim)
    tlag = Tlag(obs, sim)
    # Combine all metrics into a DataFrame for reference
    metrics = {
        'Bias': bias,
        'PBias': pbias,
        'RMSE': rmse,
        'CC': cc,
        'NSE': nse,
        'KGE': kge,
        'MRE': mre,
        'Tlag': tlag
    }
    metrics_df = pd.DataFrame(metrics)
    return metrics_df
