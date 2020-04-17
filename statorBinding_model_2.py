# kin. montecarlo simulation of stator binding and unbinding 
# in 3 states system (Unbound, Weakly-bound, Strongly-bound) for catch bond
#
#   U  --kuw-->  W  --kws--> S  --ksu--> U
#     <--kwu--      <--ksw--   <--kus--
#
# where likely kus=0 ksu=eps, N.bound = Ns + Nw.
# a catch bond would give kws(Force), ksw(Force)



import sys
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from scipy.optimize import curve_fit                                                    



# max numb. of stators:
Nmax = 14

def KMC3states_maketrace(kuw=1, kus=0, kwu=1, kws=1, ksu=0, ksw=1, Ns0=0, Nw0=0, Npts=1e3, plots=0, plots_all=False, clear_plot=True, color='k'):
    ''' Kinetic Montecarlo 
    dNs/dt = -ksu*Ns -ksw*Ns +kws*Nw +kus*(Nmax-Ns-Nw)  
    dNw/dt = -kwu*Nw +ksw*Ns -kws*Nw +kuw*(Nmax-Ns-Nw)  
    '''
    global Nmax
    kuw, kus = float(kuw), float(kus)
    kws, kwu = float(kws), float(kwu)
    ksw, ksu = float(ksw), float(ksu)
    Npts = int(Npts)
    # strongly bound stators:
    Ns = np.zeros(Npts)
    # weakly bound stators:
    Nw = np.zeros(Npts)
    # unbound stators (just to keep track, not used):
    Nu = np.zeros(Npts)
    # initial conditions:
    Ns[0] = Ns0
    Nw[0] = Nw0
    time = np.zeros(Npts)
    # (0,1] uniform random number for next state:
    ra1 = np.random.rand(Npts)
    # (0,1] uniform random number for time step:
    ra2 = np.random.rand(Npts)
    for t in range(Npts-1):
        Rsu = ksu*Ns[t]
        Rsw = ksw*Ns[t]
        Rws = kws*Nw[t]
        Rwu = kwu*Nw[t]
        Rus = kus*(Nmax-Ns[t]-Nw[t])
        Ruw = kuw*(Nmax-Ns[t]-Nw[t])
        Rtot    = [Rsu, Rsw, Rws, Rwu, Rus, Ruw]
        Ns_next = [-1 , -1 ,  1 ,  0 ,  1 ,  0 ]
        Nw_next = [ 0 ,  1 , -1 , -1 ,  0 ,  1 ]
        Nu_next = [ 1 ,  0 ,  0 ,  1 , -1 , -1 ]
        # cumulative rate array:
        rvec = np.cumsum(Rtot)
        # dice for next state (idx of rvec_next):
        choose = np.where(rvec/rvec[-1] > ra1[t])[0][0]
        # update state Ns Nw (Nu):
        Ns[t+1] = Ns[t] + Ns_next[choose]
        Nw[t+1] = Nw[t] + Nw_next[choose]
        Nu[t+1] = Nu_next[choose]
        # update time:
        time[t+1] = time[t] + np.log(1/ra2[t])/rvec[-1]
    if plots or plots_all: 
        # add fake noise for plot:
        #Ns = Ns + np.random.randn(Npts)*0.0
        #Nw = Nw + np.random.randn(Npts)*0.0
        plt.figure('KMC3states_maketrace')     
        if clear_plot: plt.clf()
        if plots_all:
            #plt.plot(time, Nu, 'y-o' , ms=2, lw=3, alpha=0.3, label='Nu')
            plt.plot(time, Ns, 'b-o', ms=2, drawstyle='steps-post', lw=3, alpha=0.3, label='Ns')
            plt.plot(time, Nw, 'r-o', ms=2, drawstyle='steps-post', lw=3, alpha=0.3, label='Nw')
        plt.plot(time, Ns+Nw, '-o', color=color, ms=3, drawstyle='steps-post', label='Ns+Nw',alpha=0.5)
        plt.legend()
        plt.grid(True)
        plt.ylabel('N(t)')
        plt.xlabel('Time (s)')
        plt.ylim(0,Nmax+1)
    return time,Ns,Nw







def resurr_stall_avg(kuw=1, kus=0, kwu=1, kws=1, ksu=0, ksw=1, Ns0_stal=0, Nw0_stal=Nmax, Ns0_res=0, Nw0_res=0, navg=100, Npts=100, nbins=100, plots=True, plots_all=False):
    '''run KMC3states_maketrace() 'navg' times,
    for resurrection and stall-release and find the average N,
    binning on nbins over time'''
    N_res = []
    N_res1 = []
    N_res2 = []
    t_res = []
    N_sta = []
    N_sta1 = []
    N_sta2 = []
    t_sta = []
    dwt_0 = []
    dwt_1 = []
    dwt_2 = []
    if plots or plots_all:
        fig1= plt.figure('resurr_stall_avg()', clear=True)
        ax1 = fig1.add_subplot(111)
        fig0 = plt.figure('resurr_stall_avg (w+s)', clear=True)
        ax0 = fig0.add_subplot(111)
        fig2 = plt.figure('resurr_stall_avg (w)', clear=True)
        ax2 = fig2.add_subplot(111)
        fig3 = plt.figure('dwell times', clear=True)
        ax3 = fig3.add_subplot(111)
    for i in range(navg):
        print(f'{i}/{navg}', end='\r')
        # resurrection: 
        _t_res, _Ns_res, _Nw_res = KMC3states_maketrace(kuw=kuw, kus=kus, kwu=kwu, kws=kws, ksu=ksu, ksw=ksw, Ns0=Ns0_res, Nw0=Nw0_res, Npts=Npts, plots=0)
        # stall-release:
        _t_sta, _Ns_sta, _Nw_sta = KMC3states_maketrace(kuw=kuw, kus=kus, kwu=kwu, kws=kws, ksu=ksu, ksw=ksw, Ns0=Ns0_stal, Nw0=Nw0_stal, Npts=Npts, plots=0)
        # times: 
        t_res = np.append(t_res, _t_res)
        t_sta = np.append(t_sta, _t_sta)
        # if both strongly and weakly bound are measurable:
        N_res = np.append(N_res, _Ns_res + _Nw_res)
        N_sta = np.append(N_sta, _Ns_sta + _Nw_sta)
        # if only strongly bound is measurable:
        N_res1 = np.append(N_res1, _Ns_res)
        N_sta1 = np.append(N_sta1, _Ns_sta)
        # if only weakly bound is measurable:
        N_res2 = np.append(N_res2, _Nw_res)
        N_sta2 = np.append(N_sta2, _Nw_sta)
        # dwell times:
        dwt_0 = np.append(dwt_0, find_dwelltimes(t_res, _Ns_res+_Nw_res))
        dwt_1 = np.append(dwt_0, find_dwelltimes(t_res, _Ns_res))
        dwt_0 = np.append(dwt_0, find_dwelltimes(t_res, _Nw_res))
        if plots_all:
            ax0.plot(_t_res, _Nw_res + _Ns_res + 0.2, 'g.', ms=2, alpha=.1)
            ax0.plot(_t_sta, _Nw_sta + _Ns_sta, 'bs', ms=2, alpha=.1)
            ax2.plot(_t_res, _Nw_res + 0.2, 'g.', ms=2, alpha=.1)
            ax2.plot(_t_sta, _Nw_sta, 'b.', ms=2, alpha=.1)

    tm_sta,  Nm_sta  = findAverageTrace(t_sta, N_sta,  bins=nbins) 
    tm_res,  Nm_res  = findAverageTrace(t_res, N_res,  bins=nbins) 
    tm_sta1, Nm_sta1 = findAverageTrace(t_sta, N_sta1, bins=nbins) 
    tm_res1, Nm_res1 = findAverageTrace(t_res, N_res1, bins=nbins) 
    tm_sta2, Nm_sta2 = findAverageTrace(t_sta, N_sta2, bins=nbins) 
    tm_res2, Nm_res2 = findAverageTrace(t_res, N_res2, bins=nbins) 
    # 2 states Langmuir fit (cut ends):
    tm_sta_c  =  tm_sta[:int(len(tm_sta)*0.8)]
    Nm_sta_c  =  Nm_sta[:int(len(Nm_sta)*0.8)]
    tm_res_c  =  tm_res[:int(len(tm_res)*0.8)]
    Nm_res_c  =  Nm_res[:int(len(Nm_res)*0.8)]
    tm_sta1_c = tm_sta1[:int(len(tm_sta1)*0.8)]
    Nm_sta1_c = Nm_sta1[:int(len(Nm_sta1)*0.8)]
    tm_res1_c = tm_res1[:int(len(tm_res1)*0.8)]
    Nm_res1_c = Nm_res1[:int(len(Nm_res1)*0.8)]
    tm_sta2_c = tm_sta2[:int(len(tm_sta2)*0.8)]
    Nm_sta2_c = Nm_sta2[:int(len(Nm_sta2)*0.8)]
    tm_res2_c = tm_res2[:int(len(tm_res2)*0.8)]
    Nm_res2_c = Nm_res2[:int(len(Nm_res2)*0.8)]
    popt_sta, _  = curve_fit(Langmuir, tm_sta_c,  Nm_sta_c,  p0=(0.01,0.01,Nmax), bounds=(0,np.inf))
    popt_res, _  = curve_fit(Langmuir, tm_res_c,  Nm_res_c,  p0=(0.01,0.01,0)   , bounds=(0,np.inf))
    popt_sta1, _ = curve_fit(Langmuir, tm_sta1_c, Nm_sta1_c, p0=(0.01,0.01,Nmax), bounds=(0,np.inf))
    popt_res1, _ = curve_fit(Langmuir, tm_res1_c, Nm_res1_c, p0=(0.01,0.01,0)   , bounds=(0,np.inf))
    popt_sta2, _ = curve_fit(Langmuir, tm_sta2_c, Nm_sta2_c, p0=(0.01,0.01,Nmax), bounds=(0,np.inf))
    popt_res2, _ = curve_fit(Langmuir, tm_res2_c, Nm_res2_c, p0=(0.01,0.01,0)   , bounds=(0,np.inf))
    
    if plots or plots_all:
        ax1.plot(tm_res,  Nm_res,  'go',  ms=4, alpha=.2, label='res s+w')
        ax1.plot(tm_sta,  Nm_sta,  'gs',  ms=4, alpha=.2, label='sta s+w')
        ax1.plot(tm_res1, Nm_res1, 'ro',  ms=4, alpha=.2, label='res s')
        ax1.plot(tm_sta1, Nm_sta1, 'rs',  ms=4, alpha=.2, label='sta s')
        ax1.plot(tm_res2, Nm_res2, 'bo',  ms=4, alpha=.2, label='res w')
        ax1.plot(tm_sta2, Nm_sta2, 'bs',  ms=4, alpha=.2, label='sta w')
        ax1.plot(tm_sta,  Langmuir(tm_sta,  popt_sta[0],  popt_sta[1],  popt_sta[2]),  '--g', lw=2)
        ax1.plot(tm_res,  Langmuir(tm_res,  popt_res[0],  popt_res[1],  popt_res[2]),  '--g', lw=2)
        ax1.plot(tm_sta1, Langmuir(tm_sta1, popt_sta1[0], popt_sta1[1], popt_sta1[2]), '--r', lw=2)
        ax1.plot(tm_res1, Langmuir(tm_res1, popt_res1[0], popt_res1[1], popt_res1[2]), '--r', lw=2)
        ax1.plot(tm_sta2, Langmuir(tm_sta2, popt_sta2[0], popt_sta2[1], popt_sta2[2]), '--b', lw=2)
        ax1.plot(tm_res2, Langmuir(tm_res2, popt_res2[0], popt_res2[1], popt_res2[2]), '--b', lw=2)
        ax1.set_ylim(0,Nmax)
        ax1.set_xlabel('Time(s)')
        ax1.legend()

        ax0.plot(tm_res, Nm_res, 'go',  ms=4, alpha=.4, label='resur.')
        ax0.plot(tm_sta, Nm_sta, 'bo',  ms=4, alpha=.4, label='stall')
        ax0.plot(tm_sta, Langmuir(tm_sta, popt_sta[0], popt_sta[1], popt_sta[2]), '--b', lw=2)
        ax0.plot(tm_res, Langmuir(tm_res, popt_res[0], popt_res[1], popt_res[2]), '--g', lw=2)
        title1 = 'kuw={:.2f}, kus={:.2f}, kwu={:.2f}, kws={:.2f}, ksu={:.2f}, ksw={:.2f},'.format(kuw,kus,kwu,kws,ksu,ksw)
        title2 = '(W+S) resur_tc = {:.2f}    stall_tc = {:.2f}    t_res/t_sta = {:.2f}'.format(1./(popt_res[0]+popt_res[1]), 1./(popt_sta[0]+popt_sta[1]), (1./(popt_res[0]+popt_res[1])) / (1./(popt_sta[0]+popt_sta[1])))
        ax0.set_title(title1+'\n'+title2)
        ax0.set_xlabel('Time(s)')
        ax0.set_ylabel('N')
        ax0.legend()
        ax0.set_ylim(0,Nmax)
        plt.tight_layout()

        ax2.plot(tm_res2, Nm_res2, 'go',  ms=4, alpha=.4, label='resur.')
        ax2.plot(tm_sta2, Nm_sta2, 'bo',  ms=4, alpha=.4, label='stall')
        ax2.plot(tm_sta2, Langmuir(tm_sta2, popt_sta2[0], popt_sta2[1], popt_sta2[2]), '--b', lw=2)
        ax2.plot(tm_res2, Langmuir(tm_res2, popt_res2[0], popt_res2[1], popt_res2[2]), '--g', lw=2)
        title1 = 'kuw={:.2f}, kus={:.2f}, kwu={:.2f}, kws={:.2f}, ksu={:.2f}, ksw={:.2f},'.format(kuw,kus,kwu,kws,ksu,ksw)
        title2 = '(W) resur_tc = {:.2f}    stall_tc = {:.2f}    t_res/t_sta = {:.2f}'.format(1./(popt_res2[0]+popt_res2[1]), 1./(popt_sta2[0]+popt_sta2[1]), (1./(popt_res2[0]+popt_res2[1])) / (1./(popt_sta2[0]+popt_sta2[1])))
        ax2.set_title(title1+'\n'+title2)
        ax2.set_xlabel('Time(s)')
        ax2.set_ylabel('N')
        ax2.legend()
        ax2.set_ylim(0,Nmax)
        plt.tight_layout()

        bbins = np.logspace(np.log10(np.min(dwt_0)), np.log10(np.max(dwt_0)), 100)
        ax3.hist(dwt_0, bins=bbins)

    print('W+S Langmuir: resur_tc = {:.2f}\t stall_tc = {:.2f}\t t_res/t_sta = {:.2f}'.format(1./(popt_res[0]+popt_res[1]), 1./(popt_sta[0]+popt_sta[1]), (1./(popt_res[0]+popt_res[1])) / (1./(popt_sta[0]+popt_sta[1]))))
    print('W   Langmuir: resur_tc = {:.2f}\t stall_tc = {:.2f}\t t_res/t_sta = {:.2f}'.format(1./(popt_res1[0]+popt_res1[1]), 1./(popt_sta1[0]+popt_sta1[1]), (1./(popt_res1[0]+popt_res1[1])) / (1./(popt_sta1[0]+popt_sta1[1]))))

    # plot single trace example resur:
    KMC3states_maketrace(kuw=kuw, kus=kus, kwu=kwu, kws=kws, ksu=ksu, ksw=ksw, Ns0=0,    Nw0=0, Npts=Npts, plots=True, plots_all=True, clear_plot=True, color='k')
    KMC3states_maketrace(kuw=kuw, kus=kus, kwu=kwu, kws=kws, ksu=ksu, ksw=ksw, Ns0=0, Nw0=Nmax, Npts=Npts, plots=True, plots_all=True, clear_plot=False, color='g')

    return dwt_0, dwt_1, dwt_2


    
def Langmuir(t, kon, koff, N0):
    ''' exponential solution of simple 2 states Langmuir model '''
    Nmax = 12.
    kon = float(kon)
    koff = float(koff)
    N0 = float(N0)
    Neq = Nmax/(1. + (koff/kon))
    N = Neq + (N0-Neq)*np.exp(-(kon+koff)*t)
    return N



def findAverageTrace(ts, Ns, bins=100):
    ''' find the average N(t) binning the ts, and averaging the Ns '''
    T = np.linspace(np.min(ts), np.max(ts), bins)
    Ns_mn = []
    ts_mn = []
    for i in range(len(T)-1):
        idx = np.nonzero((T[i] < ts) & (ts < T[i+1]))[0]
        Ns_mn = np.append(Ns_mn, np.mean(Ns[idx]))
        ts_mn = np.append(ts_mn, np.mean(ts[idx]))
    return ts_mn, Ns_mn

    

def find_dwelltimes(time, sig):
    '''find dwell times in trace N(t) '''
    #print(time[np.nonzero(np.diff(sig))[0] + 1])
    return np.diff(time[np.nonzero(np.diff(sig))[0] + 1])
        





