#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:26:09 2020
@author: vaulthigh
The first version of this script was BergStatorModel.py

Note to user:  The following models contain a parameter that is currently 
set by hand, setting the intial conditions of resurrection, and are marked 
with a TODO:  TwoStateCatchSimple, TwoStateCatchTrad, Yuan2019, Intermediate, 
TriangleIntermediate
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from numpy import random
from scipy.optimize import curve_fit, minimize, brute, basinhopping
from curve_fit import annealing

FPS=1000.
Nmax = 13.

def GlobalFitRateConstants(D, model='PNAS2017', fit_release=1, fit_steadystate=1, bootstraps=1, plots=1):
    
    _,_,_,_,_,_,N_before,N_after,N_resurrection = Dict2DataMatrix(D)
    N_before[N_before==0] = np.nan
    N_after[N_after==0] = np.nan
    time_before = np.arange(len(N_before[0,:]))/FPS
    time_after = np.arange(len(N_after[0,:]))/FPS
    time_resurrection = np.arange(len(N_resurrection[0,:]))/FPS 
    #Stop the resurrection fitting after 660s
    b = int(660*FPS)
    numtraces,_ = np.shape(N_before)
    
    N_before_avg, N_after_avg, N_resurrection_avg = np.array(np.nanmean(N_before,0)),np.array(np.nanmean(N_after,0)),np.array(np.nanmean(N_resurrection,0))

    if model=='PNAS2017':
        popt_N, popt_N_all = np.ones((bootstraps,4))* np.nan, np.ones((bootstraps,4))* np.nan
        param_guess = [0.01, 0.01, np.nanmean(N_after_avg[:100]), 0]
        bounds = ((.00001,1),(.00001,1),(np.nanmean(N_after_avg[:100])-.01,np.nanmean(N_after_avg[:100])+.01),(-.01,.01))
    elif model=='Berg2019': 
        popt_N, popt_N_all= np.ones((bootstraps,5))* np.nan, np.ones((bootstraps,5))* np.nan
        param_guess = [.008, 200, -.4, np.nanmean(N_after_avg[:100]), 0]
        bounds = ((.00005,.3),(.1,500),(-4,4),(np.nanmean(N_after_avg[:100])-.01,np.nanmean(N_after_avg[:100])+.01),(-.01,.01))
    elif model in ['TwoStateCatchSimple','Yuan2019','Intermediate']:
        popt_N, popt_N_all = np.ones((bootstraps,6))* np.nan, np.ones((bootstraps,6))* np.nan
        #param_guess = [.01, .01, .003, .003, np.nanmean(N_after_avg[:100]), 0]
        param_guess = [.01, .01, .01, .001, np.nanmean(N_after_avg[:100]), 0]
        bounds = ((.00001,10),(.00001,10),(.00001,10),(.00001,10),(np.nanmean(N_after_avg[:100])-.01,np.nanmean(N_after_avg[:100])+.01),(-.01,.01)) 
    elif model in ['TwoStateCatchTrad', 'TriangleIntermediate']:
        popt_N, popt_N_all = np.ones((bootstraps,8))* np.nan, np.ones((bootstraps,8))* np.nan
        param_guess = [.008, .007, .002, .002, .0005, .0001, np.nanmean(N_after_avg[:100]), 0]
        bounds = ((.0001,1),(.0001,1),(.0001,1),(.0001,1),(.000001,.1),(.000001,.1),(np.nanmean(N_after_avg[:100])-.01,np.nanmean(N_after_avg[:100])+.01),(-.01,.01))
    else:
        print('Model not recognized.')
        
    minimizer_kwargs = { "method": "L-BFGS-B","bounds":bounds, 'args':([time_before,time_after,time_resurrection[:b]],[N_before_avg, N_after_avg, N_resurrection_avg[:b]],model,fit_release,fit_steadystate) }
    out = basinhopping(globalfitting,param_guess,niter=8,minimizer_kwargs=minimizer_kwargs)
    param_guess[:-2] = out.x[:-2] 
    print(param_guess)
    
    out = minimize(globalfitting,param_guess,args=([time_before,time_after,time_resurrection[:b][N_resurrection_avg[0:b]>0]],[N_before_avg,N_after_avg, N_resurrection_avg[:b][N_resurrection_avg[0:b]>0]],model,fit_release,fit_steadystate),bounds=bounds, method='L-BFGS-B')
    popt_N = out.x
    print(popt_N)
        
    for n in range(bootstraps):
        print('bootstrap ', n)
        deck = np.arange(numtraces)
        random.shuffle(deck)
        idx = deck[:int(numtraces*.93)-1]
        N_before_avg, N_after_avg, N_resurrection_avg = np.array(np.nanmean(N_before[idx],0)),np.array(np.nanmean(N_after[idx],0)),np.array(np.nanmean(N_resurrection[idx],0))     
        out = minimize(globalfitting,param_guess,args=([time_before,time_after,time_resurrection[:b][N_resurrection_avg[0:b]>0]],[N_before_avg,N_after_avg, N_resurrection_avg[:b][N_resurrection_avg[0:b]>0]],model,fit_release,fit_steadystate),bounds=bounds, method='L-BFGS-B')
        popt_N_all[n,:] = out.x
        
    if plots:
        PlotStuff(D,model,fit_release,fit_steadystate, popt_N, popt_N_all)
        
    return



######### MODELS #########

def PNAS2017(x, *params, **kwarg):
    ''' 
    Langmuir model, used in Nord PNAS 2017
    par=[kon,koff,N0]
    '''
    if kwarg:
        params = kwarg['par']
    if not isinstance(params[0],float):
        params=params[0]
        
    kon = float(params[0])
    koff = float(params[1])
    N0 = float(params[2])
    
    Neq = Nmax/(1. + (koff/kon))
    N = Neq + (N0-Neq)*np.exp(-(kon+koff)*x)
    return N


def Berg2019(t, *params, **kwarg):
    ''' 
    model from Berg PNAS 2019
    par=[k0,alpha,beta,N0]
    '''
    if not isinstance(params[0],float):
        params=params[0]
        
    k0 = float(params[0])
    alpha = float(params[1])
    beta = float(params[2])
    N0 = float(params[3])
    
    # function that returns dy/dt
    def Berg2019Model(N,t,k0,alpha,beta):
        kon = k0*(1-np.exp(-alpha/N))
        koff = kon*np.exp(beta)
        dNdt = kon*(Nmax-N)-koff*N
        return dNdt
    
    # solve ODEs
    N = odeint(Berg2019Model,N0,t,args=(k0,alpha,beta))
    
    N = N[:,0]
    return N


def TwoStateCatchSimple(t, *params):
    ''' 
    simplified two state catch bond model, with a weak and strong 
    torque-producing state. par=[kuw,kwu,kws,ksw,N0]
    '''
    if not isinstance(params[0],float):
        params=params[0]
        
    k_uw = float(params[0])
    k_wu = float(params[1])
    k_ws = float(params[2])
    k_sw = float(params[3])
    N0 = float(params[4])
    #TODO: initial cond can be changed here, not yet a coded param.
    WS = [N0*.5,N0 - N0*.5]
    
    def TwoStateCatchSimpleModel(WS,x,k_uw,k_wu,k_ws,k_sw):
        W,S = WS
        dWdt = k_uw*(Nmax-W-S) - (k_wu+k_ws)*W + k_sw*S 
        dSdt = k_ws*W - k_sw*S
        return [dWdt, dSdt]

    sol = odeint(TwoStateCatchSimpleModel,WS,t,args=(k_uw,k_wu,k_ws,k_sw))
    W,S = sol[:,0], sol[:,1]
    
    N = W + S
    
    return N


def TwoStateCatchTrad(t, *params):
    ''' 
    full two state catch bond model, with a weak and strong 
    torque-producing state. par=[kuw,kwu,kws,ksw,kus,ksu,N0]
    '''
    if not isinstance(params[0],float):
        params=params[0]
        
    k_uw = float(params[0])
    k_wu = float(params[1])
    k_ws = float(params[2])
    k_sw = float(params[3])
    k_us = float(params[4])
    k_su = float(params[5])
    N0 = float(params[6])
    #TODO: initial cond can be changed here, not yet a coded param.
    WS = [N0*.5,N0 - N0*.5]
    
    def TwoStateCatcTradModel(WS,x,k_uw,k_wu,k_ws,k_sw,k_us,k_su):
        W,S = WS
        dWdt = k_uw*(Nmax-W-S) - (k_wu+k_ws)*W + k_sw*S 
        dSdt = k_us*(Nmax-W-S) - (k_sw+k_su)*S + k_ws*W
        return [dWdt, dSdt]
    
    sol = odeint(TwoStateCatcTradModel,WS,t,args=(k_uw,k_wu,k_ws,k_sw,k_us,k_su))
    W,S = sol[:,0], sol[:,1]
    
    N = W + S
    
    return N

def Yuan2019(t, *params):
    ''' 
    hidden state model, from Yuan 2019
    par=[k_ub,k_bu,k_bi,k_ib,N0]
    '''
    if not isinstance(params[0],float):
        params=params[0]
        
    k_ub = float(params[0])
    k_bu = float(params[1])
    k_bi = float(params[2])
    k_ib = float(params[3])
    N0 = float(params[4])
    #TODO: initial cond can be changed here, not yet a coded param.
    if N0<1:
        BI = [0,4]
    else:
        BI = [N0,0]
    
    def Yuan2019Model(BI,x,k_ub,k_bu,k_bi,k_ib):
        B,I = BI
        dBdt = k_ub*(Nmax-B-I) - (k_bu+k_bi)*B + k_ib*I 
        dIdt = k_bi*B - k_ib*I
        return [dBdt, dIdt]

    sol = odeint(Yuan2019Model,BI,t,args=(k_ub,k_bu,k_bi,k_ib))
    B,I = sol[:,0], sol[:,1]
    
    N = B 
    
    return N

def Intermediate(t, *params):
    ''' 
    alternative hidden state model, where the hidden state is between
    U and B. par=[k_ui,k_iu,k_ib,k_bi,N0]
    '''
    if not isinstance(params[0],float):
        params=params[0]
        
    k_ui = float(params[0])
    k_iu = float(params[1])
    k_ib = float(params[2])
    k_bi = float(params[3])
    N0 = float(params[4])
    #TODO: initial cond can be changed here, not yet a coded param.
    if N0<1:
        IB = [10,0]
    else:
        IB = [0,N0]
    
    def IntermediateModel(IB,x,k_ui,k_iu,k_ib,k_bi):
        I,B = IB
        dIdt = k_ui*(Nmax-I-B) - (k_iu+k_ib)*I + k_bi*B 
        dBdt = k_ib*I - k_bi*B
        return [dIdt, dBdt]

    sol = odeint(IntermediateModel,IB,t,args=(k_ui,k_iu,k_ib,k_bi))
    I,B = sol[:,0], sol[:,1]
    
    N = B 
    
    return N

def TriangleIntermediate(t, *params):
    ''' 
    fully connected hidden state model.
    par=[k_ui,k_iu,k_ib,k_bi,k_ub,k_bu,N0]
    '''
    if not isinstance(params[0],float):
        params=params[0]
        
    k_ui = float(params[0])
    k_iu = float(params[1])
    k_ib = float(params[2])
    k_bi = float(params[3])
    k_ub = float(params[4])
    k_bu = float(params[5])
    N0 = float(params[6])
    #TODO: initial cond can be changed here, not yet a coded param.
    if N0<1:
        IB = [4,0]
    else:
        IB = [0,N0]
    
    def TriangleIntermediateModel(IB,x,k_ui,k_iu,k_ib,k_bi,k_ub,k_bu):
        I,B = IB
        dIdt = k_ui*(Nmax-I-B) - (k_iu+k_ib)*I + k_bi*B 
        dBdt = k_ub*(Nmax-I-B) - (k_bi+k_bu)*B + k_ib*I
        return [dIdt, dBdt]
    
    sol = odeint(TriangleIntermediateModel,IB,t,args=(k_ui,k_iu,k_ib,k_bi,k_ub,k_bu))
    I,B = sol[:,0], sol[:,1]
    
    N = B
    
    return N

#############################################

def PlotStuff(D,model,fit_release, fit_steadystate, popt_N, popt_N_all):
    
    _,_,_,_,_,_,N_before,N_after,N_resurrection = Dict2DataMatrix(D)
    N_before[N_before==0] = np.nan
    N_after[N_after==0] = np.nan
    time_before= (np.arange(len(N_before[0,:])) - len(N_before[0,:]) - 1)/FPS
    time_after = np.arange(len(N_after[0,:]))/FPS
    time_resurrection = np.arange(len(N_resurrection[0,:]))/FPS 
    c = 100
    a = int(len(N_after[0,:])*1.)
    #Stop the resurrection fitting after 660s
    b = int(660*FPS)
    N_before_avg, N_after_avg, N_resurrection_avg = np.array(np.nanmean(N_before,0)),np.array(np.nanmean(N_after,0)),np.array(np.nanmean(N_resurrection,0))
    N_before_std, N_after_std, N_resurrection_std = np.array(np.nanstd(N_before,0)),np.array(np.nanstd(N_after,0)),np.array(np.nanstd(N_resurrection,0))

    
    if model=='PNAS2017':
        kon, koff = popt_N[0], popt_N[1]
        kon_std, koff_std = np.std(popt_N_all,axis=0)[0], np.std(popt_N_all,axis=0)[1]
    if model=='Berg2019':
        k0, alpha, beta = popt_N[0], popt_N[1],popt_N[2]
        k0_std, alpha_std, beta_std = np.std(popt_N_all,axis=0)[0],np.std(popt_N_all,axis=0)[1],np.std(popt_N_all,axis=0)[2]
    if model=='TwoStateCatchSimple':
        k_uw, k_wu, k_ws, k_sw = popt_N[0], popt_N[1],popt_N[2], popt_N[3]
        k_uw_std, k_wu_std, k_ws_std, k_sw_std = np.std(popt_N_all,axis=0)[0],np.std(popt_N_all,axis=0)[1],np.std(popt_N_all,axis=0)[2],np.std(popt_N_all,axis=0)[3]
    if model=='TwoStateCatchTrad':
        k_uw, k_wu, k_ws, k_sw, k_us, k_su = popt_N[0], popt_N[1],popt_N[2], popt_N[3], popt_N[4], popt_N[5]
        k_uw_std, k_wu_std, k_ws_std, k_sw_std,k_us_std, k_su_std = np.std(popt_N_all,axis=0)[0],np.std(popt_N_all,axis=0)[1],np.std(popt_N_all,axis=0)[2],np.std(popt_N_all,axis=0)[3],np.std(popt_N_all,axis=0)[4],np.std(popt_N_all,axis=0)[5]
    if model=='Yuan2019':
        k_ub, k_bu, k_bi, k_ib = popt_N[0], popt_N[1],popt_N[2], popt_N[3]
        k_ub_std, k_bu_std, k_bi_std, k_ib_std = np.std(popt_N_all,axis=0)[0],np.std(popt_N_all,axis=0)[1],np.std(popt_N_all,axis=0)[2],np.std(popt_N_all,axis=0)[3]
    if model=='Intermediate':
        k_ui, k_iu, k_ib, k_bi = popt_N[0], popt_N[1],popt_N[2], popt_N[3]
        k_ui_std, k_iu_std, k_ib_std, k_bi_std = np.std(popt_N_all,axis=0)[0],np.std(popt_N_all,axis=0)[1],np.std(popt_N_all,axis=0)[2],np.std(popt_N_all,axis=0)[3]
    if model=='TriangleIntermediate':
        k_ui, k_iu, k_ib, k_bi, k_ub, k_bu = popt_N[0], popt_N[1],popt_N[2], popt_N[3], popt_N[4], popt_N[5]
        k_ui_std, k_iu_std, k_ib_std, k_bi_std, k_ub_std, k_bu_std = np.std(popt_N_all,axis=0)[0],np.std(popt_N_all,axis=0)[1],np.std(popt_N_all,axis=0)[2],np.std(popt_N_all,axis=0)[3],np.std(popt_N_all,axis=0)[4],np.std(popt_N_all,axis=0)[5]
        
    plt.figure()
    plt.plot([time_before[-1],time_resurrection[-1]+time_after[-1]],np.ones(2)*np.mean(N_before_avg),'b--',alpha=.3)
    plt.fill_between(time_before[::c],N_before_avg[::c]-N_before_std[::c],N_before_avg[::c]+N_before_std[::c],color='k',alpha=.3)
    plt.plot(time_before[::c],N_before_avg[::c],'k',lw=3)
    plt.fill_between(time_after[::c],N_after_avg[::c]-N_after_std[::c],N_after_avg[::c]+N_after_std[::c],color='k',alpha=.3)
    plt.plot(time_after[::c],N_after_avg[::c],'k',lw=3)
    plt.fill_between(time_resurrection[::c]+ time_after[-1],N_resurrection_avg[::c]-N_resurrection_std[::c],N_resurrection_avg[::c]+N_resurrection_std[::c],color='k',alpha=.3)
    plt.plot(time_resurrection[::c]+ time_after[-1],N_resurrection_avg[::c],'k',lw=3)
    if fit_release:
        if model=='PNAS2017':
            plt.plot(time_after[0:a:c],PNAS2017(time_after[0:a:c], kon, koff, popt_N[2]),'r-')
        elif model=='Berg2019':
            plt.plot(time_after[0:a:c],Berg2019(time_after[0:a:c], k0,alpha,beta, popt_N[3]),'r-')
        elif model=='TwoStateCatchSimple':
            plt.plot(time_after[0:a:c],TwoStateCatchSimple(time_after[0:a:c], k_uw,k_wu,k_ws,k_sw, popt_N[4]),'r-')
        elif model=='TwoStateCatchTrad':
            plt.plot(time_after[0:a:c],TwoStateCatchTrad(time_after[0:a:c], k_uw,k_wu,k_ws,k_sw,k_us,k_su, popt_N[6]),'r-')
        elif model=='Yuan2019':
            plt.plot(time_after[0:a:c],Yuan2019(time_after[0:a:c], k_ub,k_bu,k_bi,k_ib, popt_N[4]),'r-')
        elif model=='Intermediate':
            plt.plot(time_after[0:a:c],Intermediate(time_after[0:a:c], k_ui,k_iu,k_ib,k_bi, popt_N[4]),'r-')
        elif model=='TriangleIntermediate':
            plt.plot(time_after[0:a:c],TriangleIntermediate(time_after[0:a:c], k_ui,k_iu,k_ib,k_bi,k_ub,k_bu, popt_N[6]),'r-')
    if fit_steadystate:
        if model=='PNAS2017':
            plt.plot([time_before[0],time_before[-1]],np.ones(2)*PNAS2017(time_before[::c]-time_before[0], kon, koff, np.mean(N_before_avg))[-1],'r-')
        elif model=='Berg2019':
            plt.plot([time_before[0],time_before[-1]],np.ones(2)*Berg2019(time_before[::c]-time_before[0], k0,alpha,beta, np.mean(N_before_avg))[-1],'r-')
        elif model=='TwoStateCatchSimple':
            plt.plot([time_before[0],time_before[-1]],np.ones(2)*TwoStateCatchSimple(time_before[::c]-time_before[0], k_uw,k_wu,k_ws,k_sw, np.mean(N_before_avg))[-1],'r-')
        elif model=='TwoStateCatchTrad':
            plt.plot([time_before[0],time_before[-1]],np.ones(2)*TwoStateCatchTrad(time_before[::c]-time_before[0], k_uw,k_wu,k_ws,k_sw,k_us,k_su, np.mean(N_before_avg))[-1],'r-')
        elif model=='Yuan2019':
            plt.plot([time_before[0],time_before[-1]],np.ones(2)*Yuan2019(time_before[::c]-time_before[0], k_ub,k_bu,k_bi,k_ib, np.mean(N_before_avg))[-1],'r-')            
        elif model=='Intermediate':
            plt.plot([time_before[0],time_before[-1]],np.ones(2)*Intermediate(time_before[::c]-time_before[0], k_ui,k_iu,k_ib,k_bi, np.mean(N_before_avg))[-1],'r-')
        elif model=='TriangleIntermediate':
            plt.plot([time_before[0],time_before[-1]],np.ones(2)*TriangleIntermediate(time_before[::c]-time_before[0], k_ui,k_iu,k_ib,k_bi,k_ub,k_bu, np.mean(N_before_avg))[-1],'r-')
    else:
        plt.plot(time_before[::c],np.ones(len(time_before[::c]))*np.mean(N_before_avg),'b-')
        
    if model=='PNAS2017':
        plt.plot(time_resurrection[0:b:c]+ time_after[-1],PNAS2017(time_resurrection[0:b:c], kon, koff, popt_N[3]),'r-')
        plt.text(-450,0.3,'kon = %.4f +/- %.4f \nkoff = %.4f +/- %.4f' %(kon, kon_std, koff, koff_std),fontsize=14)
    if model=='Berg2019':
        plt.plot(time_resurrection[0:b:c]+ time_after[-1],Berg2019(time_resurrection[0:b:c], k0,alpha,beta, popt_N[4]),'r-')
        plt.text(-450,0.3,'k0= %.4f +/- %.4f \nalpha= %.4f +/- %.4f \nbeta= %.4f +/- %.4f' %(k0, k0_std, alpha, alpha_std, beta, beta_std),fontsize=14)
    if model=='TwoStateCatchSimple':
        plt.plot(time_resurrection[0:b:c]+ time_after[-1],TwoStateCatchSimple(time_resurrection[0:b:c], k_uw,k_wu,k_ws,k_sw, popt_N[5]),'r-')
        plt.text(-450,0.3,'kuw= %.4f +/- %.4f \nkwu= %.4f +/- %.4f\nkws= %.4f +/- %.4f\nksw= %.4f +/- %.4f' %(k_uw, k_uw_std, k_wu, k_wu_std, k_ws, k_ws_std, k_sw, k_sw_std),fontsize=14) 
    if model=='TwoStateCatchTrad':
        plt.plot(time_resurrection[0:b:c]+ time_after[-1],TwoStateCatchTrad(time_resurrection[0:b:c], k_uw,k_wu,k_ws,k_sw,k_us,k_su, popt_N[7]),'r-')
        plt.text(-450,0.3,'kuw= %.4f +/- %.4f \nkwu= %.4f +/- %.4f\nkws= %.4f +/- %.4f\nksw= %.4f +/- %.4f\nkus= %.4f +/- %.4f\nksu= %.4f +/- %.4f' %(k_uw, k_uw_std, k_wu, k_wu_std, k_ws, k_ws_std, k_sw, k_sw_std, k_us, k_us_std, k_su, k_su_std),fontsize=14) 
    if model=='Yuan2019':
        plt.plot(time_resurrection[0:b:c]+ time_after[-1],Yuan2019(time_resurrection[0:b:c], k_ub,k_bu,k_bi,k_ib, popt_N[5]),'r-')
        plt.text(-450,0.3,'kub= %.4f +/- %.4f \nkbu= %.4f +/- %.4f\nkbi= %.4f +/- %.4f\nkib= %.4f +/- %.4f' %(k_ub, k_ub_std, k_bu, k_bu_std, k_bi, k_bi_std, k_ib, k_ib_std),fontsize=14) 
    if model=='Intermediate':
        plt.plot(time_resurrection[0:b:c]+ time_after[-1],Intermediate(time_resurrection[0:b:c], k_ui,k_iu,k_ib,k_bi, popt_N[5]),'r-')
        plt.text(-450,0.3,'kui= %.4f +/- %.4f \nkiu= %.4f +/- %.4f\nkib= %.4f +/- %.4f\nkbi= %.4f +/- %.4f' %(k_ui, k_ui_std, k_iu, k_iu_std, k_ib, k_ib_std, k_bi, k_bi_std),fontsize=14) 
    if model=='TriangleIntermediate':
        plt.plot(time_resurrection[0:b:c]+ time_after[-1],TriangleIntermediate(time_resurrection[0:b:c], k_ui,k_iu,k_ib,k_bi,k_ub,k_bu, popt_N[7]),'r-')
        plt.text(-450,0.3,'kui= %.4f +/- %.4f \nkiu= %.4f +/- %.4f\nkib= %.4f +/- %.4f\nkbi= %.4f +/- %.4f\nkub= %.4f +/- %.4f\nkbu= %.4f +/- %.4f' %(k_ui, k_ui_std, k_iu, k_iu_std, k_ib, k_ib_std, k_bi, k_bi_std, k_ub, k_ub_std, k_bu, k_bu_std),fontsize=14) 
    
    plt.ylim(0,13);plt.ylabel('number stators'); plt.xlabel('time (s)')
    plt.xlim(min(time_before),1560)
    plt.ylim(0,11)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    
    return

def globalfitting(params,time,N,model,fit_release,fit_steadystate):
    #TODO: if fit steady state
    
    time_before,time_after,time_resurrection = time[0],time[1],time[2]
    N_before_avg, N_after_avg, N_resurrection_avg = N[0],N[1],N[2]
    N_ss = np.mean(N_before_avg)
    
    if model=='PNAS2017':
        k_on,k_off,N0_stall,N0_res = params[0],params[1],params[2],params[3]
        resid_resurrection= abs(N_resurrection_avg - PNAS2017(time_resurrection, k_on, k_off, N0_res))
        if fit_release:
            resid_after = abs(N_after_avg - PNAS2017(time_after, k_on, k_off, N0_stall))
        if fit_steadystate:
            resid_steadystate = abs(N_before_avg - PNAS2017(time_before, k_on, k_off, N_ss))
            
    if model=='Berg2019':
        k0,alpha,beta,N0_stall,N0_res = params[0],params[1],params[2],params[3],params[4]
        resid_resurrection= abs(N_resurrection_avg - Berg2019(time_resurrection, k0, alpha, beta, N0_res))
        if fit_release:
            resid_after = abs(N_after_avg - Berg2019(time_after, k0, alpha, beta, N0_stall))
        if fit_steadystate:
            resid_steadystate = abs(N_before_avg - Berg2019(time_before, k0, alpha, beta, N_ss))
            
    if model=='TwoStateCatchSimple':
        k_uw,k_wu,k_ws,k_sw,N0_stall,N0_res = params[0],params[1],params[2],params[3],params[4], params[5]
        resid_resurrection= abs(N_resurrection_avg - TwoStateCatchSimple(time_resurrection, k_uw, k_wu, k_ws, k_sw, N0_res))
        if fit_release:
            resid_after = abs(N_after_avg - TwoStateCatchSimple(time_after, k_uw, k_wu, k_ws, k_sw, N0_stall))
        if fit_steadystate:
            resid_steadystate = abs(N_before_avg - TwoStateCatchSimple(time_before, k_uw, k_wu, k_ws, k_sw, N_ss))
            
    if model=='TwoStateCatchTrad':
        k_uw,k_wu,k_ws,k_sw,k_us,k_su,N0_stall,N0_res = params[0],params[1],params[2],params[3],params[4], params[5],params[6], params[7]
        resid_resurrection= abs(N_resurrection_avg - TwoStateCatchTrad(time_resurrection, k_uw, k_wu, k_ws, k_sw, k_us, k_su, N0_res))
        if fit_release:
            resid_after = abs(N_after_avg - TwoStateCatchTrad(time_after, k_uw, k_wu, k_ws, k_sw, k_us, k_su, N0_stall))
        if fit_steadystate:
            resid_steadystate = abs(N_before_avg - TwoStateCatchTrad(time_before, k_uw, k_wu, k_ws, k_sw, k_us, k_su, N_ss))
    
    if model=='Yuan2019':
        k_ub,k_bu,k_bi,k_ib,N0_stall,N0_res = params[0],params[1],params[2],params[3],params[4], params[5]
        resid_resurrection= abs(N_resurrection_avg - Yuan2019(time_resurrection, k_ub,k_bu,k_bi,k_ib, N0_res))
        if fit_release:
            resid_after = abs(N_after_avg - Yuan2019(time_after, k_ub,k_bu,k_bi,k_ib, N0_stall))
        if fit_steadystate:
            resid_steadystate = abs(N_before_avg - Yuan2019(time_before, k_ub,k_bu,k_bi,k_ib, N_ss))
    
    if model=='Intermediate':
        k_ui,k_iu,k_ib,k_bi,N0_stall,N0_res = params[0],params[1],params[2],params[3],params[4], params[5]
        resid_resurrection= abs(N_resurrection_avg - Intermediate(time_resurrection, k_ui,k_iu,k_ib,k_bi, N0_res))
        if fit_release:
            resid_after = abs(N_after_avg - Intermediate(time_after, k_ui,k_iu,k_ib,k_bi, N0_stall))
        if fit_steadystate:
            resid_steadystate = abs(N_before_avg - Intermediate(time_before, k_ui,k_iu,k_ib,k_bi, N_ss))
    
    if model=='TriangleIntermediate':
        k_ui,k_iu,k_ib,k_bi,k_ub,k_bu,N0_stall,N0_res = params[0],params[1],params[2],params[3],params[4], params[5],params[6], params[7]
        resid_resurrection= abs(N_resurrection_avg - TriangleIntermediate(time_resurrection, k_ui,k_iu,k_ib,k_bi,k_ub,k_bu, N0_res))
        if fit_release:
            resid_after = abs(N_after_avg - TriangleIntermediate(time_after, k_ui,k_iu,k_ib,k_bi,k_ub,k_bu, N0_stall))
        if fit_steadystate:
            resid_steadystate = abs(N_before_avg - TriangleIntermediate(time_before, k_ui,k_iu,k_ib,k_bi,k_ub,k_bu, N_ss))
            
    Resid = resid_resurrection
    if fit_release:
        Resid = np.concatenate((resid_after, resid_resurrection), axis=None)
    if fit_steadystate:
        Resid = np.concatenate((resid_steadystate, Resid), axis=None)

    return np.nansum(Resid**2)

def Dict2DataMatrix(D):
    
    max_before = np.max([len(D[k]['torque_before_stall_pNnm']) for k in D.keys()])
    max_after = np.max([len(D[k]['torque_after_release_pNnm']) for k in D.keys()])
    max_resurrection = np.max([len(D[k]['torque_resurrection_pNnm']) for k in D.keys() if 'torque_resurrection_pNnm' in D[k].keys()])
        
    T_before = np.ones((np.max([*D.keys()])+1,max_before)) * np.nan
    T_after = np.ones((np.max([*D.keys()])+1,max_after)) * np.nan
    T_resurrection = np.ones((np.max([*D.keys()])+1,max_resurrection)) * np.nan
    S_before = np.ones((np.max([*D.keys()])+1,max_before)) * np.nan
    S_after = np.ones((np.max([*D.keys()])+1,max_after)) * np.nan
    S_resurrection = np.ones((np.max([*D.keys()])+1,max_resurrection)) * np.nan
    N_before = np.ones((np.max([*D.keys()])+1,max_before)) * np.nan
    N_after = np.ones((np.max([*D.keys()])+1,max_after)) * np.nan
    N_resurrection = np.ones((np.max([*D.keys()])+1,max_resurrection)) * np.nan
    
    for k in D.keys():
        if D[k]['use']==1:
            drag = D[k]['drag_Nms']
            sign_before = np.sign(np.mean(D[k]['torque_before_stall_pNnm']))
            sign_after = np.sign(np.mean(D[k]['torque_after_release_pNnm']))
            T_before[k,max_before-len(D[k]['torque_before_stall_pNnm']):] = D[k]['torque_before_stall_pNnm'] * sign_before
            T_after[k,:len(D[k]['torque_after_release_pNnm'])] = D[k]['torque_after_release_pNnm'] * sign_after
            S_before[k,max_before-len(D[k]['torque_before_stall_pNnm']):] = D[k]['torque_before_stall_pNnm'] / (2*np.pi*drag*1e21) * sign_before
            S_after[k,:len(D[k]['torque_after_release_pNnm'])] = D[k]['torque_after_release_pNnm']/ (2*np.pi*drag*1e21) * sign_after
            N_before[k,max_before-len(D[k]['torque_before_stall_pNnm']):] = D[k]['statnum_before_stall']
            N_after[k,:len(D[k]['torque_after_release_pNnm'])] = D[k]['statnum_after_release']
            if 'fit_speed_resurrection_Hz' in D[k].keys():
                sign_resurrection = np.sign(np.mean(D[k]['torque_resurrection_pNnm']))
                T_resurrection[k,:len(D[k]['torque_resurrection_pNnm'])] = D[k]['torque_resurrection_pNnm'] * sign_resurrection
                S_resurrection[k,:len(D[k]['torque_resurrection_pNnm'])] = D[k]['torque_resurrection_pNnm']/ (2*np.pi*drag*1e21) * sign_resurrection
                N_resurrection[k,:len(D[k]['torque_resurrection_pNnm'])] = D[k]['statnum_resurrection'] 
        
    return T_before,T_after,T_resurrection,S_before,S_after,S_resurrection,N_before,N_after,N_resurrection




