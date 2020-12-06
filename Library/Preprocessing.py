from Constants import *
import numpy as np
import pandas as pd
from scipy import optimize
from collections import defaultdict

def rank_metric(colname, data):
    mean_col = colname + '_mean'
    std_col = colname + '_std'
    rank_col = colname + '_rank'
    rank_std_up_col = colname + '_rank_std_up'
    rank_std_down_col = colname + '_rank_std_down'
    Rank_Std_up = []
    Rank_Std_down = []

    data = data.sort_values(mean_col)
    data[rank_col] = np.argsort(-1*data[mean_col])

    for i in range(len(data)):
        lower_than = np.sum((data[mean_col].iloc[i]+data[std_col].iloc[i]) >= (data[mean_col].iloc[i:]-data[std_col].iloc[i:]))-1 #-1 to not count itself
        Rank_Std_up.append(lower_than)
        greater_than = np.sum((data[mean_col].iloc[i]-data[std_col].iloc[i]) <= (data[mean_col].iloc[:i]+data[std_col].iloc[:i])) #-1 is absent because of slicing open/closed limits
        Rank_Std_down.append(greater_than)
        data['menos'] = data[mean_col]-data[std_col]
        data['mas'] = data[mean_col]+data[std_col]
    data[rank_std_up_col] = Rank_Std_up
    data[rank_std_down_col] = Rank_Std_down
    data.sort_index(inplace=True)
    return data

def ma(data, n_window):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>n_window:
            moving_ave = (cumsum[i] - cumsum[i-n_window])/n_window
            moving_aves.append(moving_ave)
    moving_aves = np.array(moving_aves)
    return moving_aves

def gaussian(x, mu, sig):
    return np.exp(-0.5*np.power((x - mu)/sig, 2.)) / (sig*(2*np.pi)**0.5)

def boltz_mean(x):
    x = np.array(x)
    RT = 0.001987 * 300 #kcal/mol
    nom = np.sum(np.multiply(np.exp(x/RT), x))
    denom = np.sum(np.exp(x/RT))
    return nom/denom
    #return np.mean(x)

def boltz_std(x):
    x = np.array(x)
    RT = 0.001987 * 300 #kcal/mol
    
    mean2 = np.sum(np.multiply(np.exp(x/RT), x**2))/np.sum(np.exp(x/RT))
    mean = np.sum(np.multiply(np.exp(x/RT), x))/np.sum(np.exp(x/RT))
    #return (np.sum(np.multiply(np.exp(x/RT), (x-mean)**2))/np.sum(np.exp(x/RT)))**0.5 #two returns equivalent
    return (mean2 - mean**2)**0.5
    #return np.std(x)/(len(x)**0.5)

def reject_outliers(data, iq=1.5):
    IQ = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    mask = np.logical_and(data<np.median(data)+iq*IQ, data>np.median(data)-iq*IQ)
    return np.array(data)[mask]

def read_file(fname):
    f = open(fname, "r")
    fl = f.readlines()
    f.close()

    clean = []
    for line in fl:
        if "#" not in line and "@" not in line:
            clean.append(line.split())
    clean = np.array(clean, dtype='float')
    return clean

def import_distances(rname):
    distances = {}
    for a1, a2 in AA_pairs:
        dist = read_file("Distances/{}/{}-{}-ASX_NVT_dcom.sfu".format(rname, a1, a2))[:,1:]
        distances["{}-{}".format(a1, a2)] = dist
    return distances

def import_radius_of_gyration(rname):
    gyration = {}
    for a1, a2 in AA_pairs:
        gyr = read_file("Gyration/{}/{}-{}-ASX_gyr.xvg".format(rname, a1, a2))[:,1]
        gyration["{}-{}".format(a1, a2)] = gyr
    return gyration

def read_dang(rname, nice_x, key, n_ignore=40):
    dang = read_file("DANGS/{}/{}-ASX_PULL_dang.sfu".format(rname, key))
    d = dang[:,:len(dang[0,:])//2]*10 # *10 because of a unit error in the dangs python script
    ang = dang[:,(len(dang[0,:])//2):]
    nice_ang = []
    for i in range(len(d[0,:])):
        angles = []
        for x in nice_x:
            closest_ndx = np.argsort(np.abs(d[:,i] - x))[:2]
            if x<np.min(d[:,i]) or x>np.max(d[:,i]):
                angles.append(-1000)
            else:
                angles.append(np.mean(ang[closest_ndx,i]))
        nice_ang.append(angles)
    nice_ang = np.array(nice_ang).T
    for i in range(len(nice_x)):
        mask = nice_ang[i,:]!=-1000
        mean = np.mean(nice_ang[i,mask])
        nice_ang[i,~mask] = mean
    return nice_ang

def calculate_rog_stats(rogs, times, tini=3, tfin=4):
    rogs_mean = np.mean(rogs[np.logical_and(times>tini, times<tfin)])
    rogs_std = np.std(rogs[np.logical_and(times>tini, times<tfin)])
    return rogs_mean, rogs_std

def read_text_file_pull(fname):
    f = open(fname, 'r')
    fl = f.readlines()
    f.close()

    data = []
    for line in fl[:-1]:
        if "@" not in line and "#" not in line:
            data.append(line.split())
    data = np.array(data, dtype='float')
    return data

def import_pullx(rname):
    pullxs = {}
    for a1, a2 in AA_pairs:
        pullx = read_text_file_pull("PULLX/{}/{}-{}-ASX_PULL_pullx.xvg".format(rname, a1, a2))
        pullxs["{}-{}".format(a1, a2)] = pullx
    return pullxs

def import_pullf(rname):
    pullfs = {}
    for a1, a2 in AA_pairs:
        pullf = read_text_file_pull("PULLF/{}/{}-{}-ASX_PULL_pullf.xvg".format(rname, a1, a2))
        pullfs["{}-{}".format(a1, a2)] = pullf
    return pullfs

def homogenize_pull(X, F, n_ignore=40, n_smooth=1000):
    n_bottle = min([len(X), len(F)])
    times = X[:n_bottle,0]
    X = X[:n_bottle,1:]
    F = F[:n_bottle,1:]

    nice_x = np.linspace(np.min(X[:,1:]), np.max(X[:,1:]), n_smooth)
    dx = nice_x[1] - nice_x[0]
    nice_f = []
    for i in range(len(X[0,:])):
        forces = []
        for x in nice_x:
            closest_ndx = np.argsort(np.abs(X[:,i] - x))[:2]
            if x<np.min(X[:,i]) or x>np.max(X[:,i]):
                forces.append(-1000)
            else:
                forces.append(np.mean(F[closest_ndx,i]))
        nice_f.append(forces)
    nice_f = np.array(nice_f).T
    for i in range(len(nice_x)):
        mask = nice_f[i,:]!=-1000
        mean = np.mean(nice_f[i,mask])
        nice_f[i,~mask] = mean

    nice_g = []
    for f in nice_f.T:
        dg = np.cumsum(f*dx)
        nice_g.append(dg)
    nice_g = np.array(nice_g).T
    nice_g = (nice_g[n_ignore:-n_ignore] - nice_g[n_ignore:-n_ignore][-1])
    return nice_x[n_ignore:-n_ignore], nice_f[n_ignore:-n_ignore]/4.184, nice_g/4.184 #kJ to kcal"""

def protein_similitude_score(data):
    max_freq = max(PDB_frequencies.values())
    max_score = 2*max_freq
    score = [0.5*(PDB_frequencies[aa1] + PDB_frequencies[aa2])/max_freq for aa1, aa2 in data[['aa1','aa2']].values]
    return score

def highlight_charged_aas(data):
    if data.aa1 in charges.keys:
        return ['background-color: gray']*len(data)
    else:
        return ['background-color: white']*len(data)


def weighting_factor(t, th, endtime=10):
    """Calculates the exponential function so that the integral under th is 0.2 and the integral until endtime is 1"""
    beta_func = lambda x: (np.exp(th*x) - 0.2*np.exp(x*endtime) - 0.8)**2
    beta = optimize.minimize(beta_func, x0=1.5).x[0]
    alpha = beta/(np.exp(beta*endtime)-1)
    weight = alpha*np.exp(beta*t)
    return weight

def calculate_weighted_fraction(distances, rog_mean, times, eta, th, endtime):
    """Weights the binding events with the exponential function from weighting_factor"""
    dt = times[-1]/len(times)
    weights = weighting_factor(times, th=th, endtime=endtime)
    n_ana = len(distances[0,:])
    bt = np.sum(distances<=eta*rog_mean, axis=1)/n_ana
    btw = np.sum(bt*weights)*dt
    btw_var = np.sum((bt-btw)**2*weights)*dt
    res = np.random.normal(loc=btw, scale=btw_var, size=n_ana)
    return bt, res

class Catecholamine_PMF:
    def __init__(self, rname):
        self.rname = rname
        pullx = import_pullx(rname)
        pullf = import_pullf(rname)
        self.keys = ["{}-{}".format(a1, a2) for a1, a2 in AA_pairs]
        self.x, self.f, self.g, self.g_bind, self.g_mean, self.g_std, self.dangs = {}, {}, {}, {}, {}, {}, {}
        for key in self.keys:
            print(key)
            self.x[key], self.f[key], self.g[key] = homogenize_pull(pullx[key], pullf[key])
            self.g_bind[key] = reject_outliers([np.min(gibbs) for gibbs in self.g[key].T], iq=1.5)
            self.g_mean[key] = boltz_mean(self.g_bind[key])
            self.g_std[key] = boltz_std(self.g_bind[key])
            self.dangs[key] = read_dang(rname, self.x[key], key)

        colnames = ['aa1', 'aa2', 'dgs', 'dg_mean', 'dg_std']
        core = [[key[:3], key[4:7], g, g_av, g_dev] \
                for key, g, g_av, g_dev \
                in zip(self.keys, self.g_bind.values(), self.g_mean.values(), self.g_std.values())]
        self.data = pd.DataFrame(core, columns=colnames)
        self.protein_similitude = protein_similitude_score(self.data)
        self.data['psc'] = self.protein_similitude

class Catecholamine_weighted:
    def __init__(self, rname, eta, th):
        self.rname = rname
        self.distances = import_distances(rname)
        self.rogs = import_radius_of_gyration(rname)
        self.rogs_mean, self.rogs_std = {}, {}
        self.times = np.linspace(0,10,self.distances['ALA-ALA'].shape[0])
        self.keys = ["{}-{}".format(a1, a2) for a1, a2 in AA_pairs]
        self.bound_hist, self.wfs, self.wf_mean, self.wf_std = {}, {}, {}, {}
        for key in self.keys:
            print(key)
            self.rogs_mean[key], self.rogs_std[key] = calculate_rog_stats(self.rogs[key], self.times)
            self.bound_hist[key], self.wfs[key] = calculate_weighted_fraction(self.distances[key], self.rogs_mean[key], self.times, eta, th, endtime=10)
            self.wf_mean[key] = np.mean(self.wfs[key])
            self.wf_std[key] = np.std(self.wfs[key])

        colnames = ['aa1', 'aa2', 'wfs', 'wf_mean', 'wf_std', 'rog_mean', 'rog_std']
        core = [[key[:3], key[4:7], wfs, wf_mean, wf_std, rog_mean, rog_std] \
                for key, wfs, wf_mean, wf_std, rog_mean, rog_std \
                in zip(self.keys, self.wfs.values(), self.wf_mean.values(), self.wf_std.values(), self.rogs_mean.values(), self.rogs_std.values())]
        self.data = pd.DataFrame(core, columns=colnames)
        self.protein_similitude = protein_similitude_score(self.data)
        self.data['psc'] = self.protein_similitude
