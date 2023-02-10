# This file has a lot of duplicated code from Mixtures/DEPENDENCIES.
# I would spend some time refactoring the code into a tidy Python package.
from Constants import *
import numpy as np
import pandas as pd
from scipy import optimize
from collections import defaultdict


def rank_metric(colname, data):
    """
    data is a dataframe with two numerical columns called colname_mean and colname_std.
    This function adds columns with the ranking (and their propagated uncertainty) according to such columns

    Return a dataframe with all the columns (old and new)
    """
    mean_col = colname + "_mean"
    std_col = colname + "_std"
    rank_col = colname + "_rank"
    rank_std_up_col = colname + "_rank_std_up"
    rank_std_down_col = colname + "_rank_std_down"
    Rank_Std_up = []
    Rank_Std_down = []

    data = data.sort_values(mean_col)
    data[rank_col] = np.argsort(-1 * data[mean_col])

    for i in range(len(data)):
        # TODO: Check that the ranking standard deviation is calculated correctly both here and in Mixtures
        # -1 to not count itself
        lower_than = (
            np.sum(
                (data[mean_col].iloc[i] + data[std_col].iloc[i])
                >= (data[mean_col].iloc[i:] - data[std_col].iloc[i:])
            )
            - 1
        )
        Rank_Std_up.append(lower_than)
        # -1 is absent because of slicing open/closed limits
        greater_than = np.sum(
            (data[mean_col].iloc[i] - data[std_col].iloc[i])
            <= (data[mean_col].iloc[:i] + data[std_col].iloc[:i])
        )
        Rank_Std_down.append(greater_than)
        data[f"{colname}_up"] = data[mean_col] - data[std_col]  # DG mean - DG std
        data[f"{colname}_down"] = data[mean_col] + data[std_col]  # DG mean + DG std
    data[rank_std_up_col] = Rank_Std_up
    data[rank_std_down_col] = Rank_Std_down
    data.sort_index(inplace=True)
    return data


def ma(data, n_window):
    """
    Calculates moving average of the 1D array data.
    n_window is the sampling's window size
    """
    cumsum, moving_aves = [0], []

    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i > n_window:
            moving_ave = (cumsum[i] - cumsum[i - n_window]) / n_window
            moving_aves.append(moving_ave)
    moving_aves = np.array(moving_aves)
    return moving_aves


def gaussian(x, mu, sig):
    """
    Returns a gaussian centered in mu and with variance sig
    x can be a 1D array or a single numeric value
    """
    return np.exp(-0.5 * np.power((x - mu) / sig, 2.0)) / (sig * (2 * np.pi) ** 0.5)


def boltz_mean(x, T=300):
    """
    Calculates Boltzmann-weighted average of the variable x at a temperature T
    """
    x = np.array(x)
    RT = 0.001987 * T  # kcal/mol
    nom = np.sum(np.multiply(np.exp(x / RT), x))
    denom = np.sum(np.exp(x / RT))
    return nom / denom


def boltz_std(x, T=300):
    """
    Calculates Boltzmann-weighted standard deviation (i.e. square root of the second statistical moment) of the variable x at a temperature T
    """
    x = np.array(x)
    RT = 0.001987 * T  # kcal/mol

    mean2 = np.sum(np.multiply(np.exp(x / RT), x**2)) / np.sum(np.exp(x / RT))
    mean = np.sum(np.multiply(np.exp(x / RT), x)) / np.sum(np.exp(x / RT))
    # return (np.sum(np.multiply(np.exp(x/RT), (x-mean)**2))/np.sum(np.exp(x/RT)))**0.5 #two returns equivalent
    return (mean2 - mean**2) ** 0.5
    # return np.std(x)/(len(x)**0.5)


def reject_outliers(data, iq=1.5):
    """
    Slices the 1D array data keeping only the values within iq times the interquartile distance from the first and third quartiles
    """
    IQ = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    mask = np.logical_and(
        data < np.median(data) + iq * IQ, data > np.median(data) - iq * IQ
    )
    return np.array(data)[mask]


def read_file(fname):
    """
    Reads text file with path fname
    Ignores lines with "#" and "@"
    Return array
    """
    f = open(fname, "r")
    fl = f.readlines()
    f.close()

    clean = []
    for line in fl:
        if "#" not in line and "@" not in line:
            clean.append(line.split())
    clean = np.array(clean, dtype="float")
    return clean


def read_dang(rname, nice_x, key, setup):
    """
    Imports angles from text file located at "DANGS/{rname}/{key}-ASX_PULL_dang.sfu"
    The function returns an array with the angles at the distances in nice_x
    """
    dang = read_file("{}/DANGS/{}/{}-ASX_PULL_dang.sfu".format(setup, rname, key))
    d = (
        dang[:, : len(dang[0, :]) // 2] * 10
    )  # *10 because of a unit error in the dangs python script
    ang = dang[:, (len(dang[0, :]) // 2) :]
    nice_ang = []
    for i in range(len(d[0, :])):
        angles = []
        for x in nice_x:
            closest_ndx = np.argsort(np.abs(d[:, i] - x))[:2]
            if x < np.min(d[:, i]) or x > np.max(d[:, i]):
                angles.append(-1000)
            else:
                angles.append(np.mean(ang[closest_ndx, i]))
        nice_ang.append(angles)
    nice_ang = np.array(nice_ang).T
    for i in range(len(nice_x)):
        mask = nice_ang[i, :] != -1000
        mean = np.mean(nice_ang[i, mask])
        nice_ang[i, ~mask] = mean
    return nice_ang


def calculate_rog_stats(rogs, times, tini=3, tfin=4):
    """
    Calculates mean and standard deviation of the 1D array rogs
    It only takes the entries with times within 3 and 4
    """
    rogs_mean = np.mean(rogs[np.logical_and(times > tini, times < tfin)])
    rogs_std = np.std(rogs[np.logical_and(times > tini, times < tfin)])
    return rogs_mean, rogs_std


def read_text_file_pull(fname):
    """
    Reads text file with path fname
    Ignores lines with "#" and "@"
    Return array
    """
    f = open(fname, "r")
    fl = f.readlines()
    f.close()

    data = []
    for line in fl[:-1]:
        if "@" not in line and "#" not in line:
            data.append(line.split())
    data = np.array(data, dtype="float")
    return data


def import_pullx(rname, setup):
    """
    Reads text file with path "{setup}/PULLX/{rname}/{aa1}-{aa2}-ASX_PULL_pullx.xvg",
    where aa1 and aa2 are values in the AA_pairs list (see Constants.py)
    Ignores lines with "#" and "@"
    Returns dictionary
    """
    pullxs = {}
    for a1, a2 in AA_pairs:
        pullx = read_text_file_pull(
            "{}/PULLX/{}/{}-{}-ASX_PULL_pullx.xvg".format(setup, rname, a1, a2)
        )
        pullxs["{}-{}".format(a1, a2)] = pullx
    return pullxs


def import_pullf(rname, setup):
    """
    Reads text file with path "{setup}/PULLF/{rname}/{aa1}-{aa2}-ASX_PULL_pullf.xvg",
    where aa1 and aa2 are values in the AA_pairs list (see Constants.py)
    Ignores lines with "#" and "@"
    Returns dictionary
    """
    pullfs = {}
    for a1, a2 in AA_pairs:
        pullf = read_text_file_pull(
            "{}/PULLF/{}/{}-{}-ASX_PULL_pullf.xvg".format(setup, rname, a1, a2)
        )
        pullfs["{}-{}".format(a1, a2)] = pullf
    return pullfs


def homogenize_pull(X, F, n_ignore=40, n_smooth=1000):
    """
    It takes arrays X and F with the position and steering force of all the analytes in a system
    It interpolates the data to obtain forces and free energies at the positions nice_x (defined below)

    n_ignore is the number of points to ignore at the beginning and the end of the positions and forces
    n_ignore is useful when some analytes are pulled away a lot more than others (allows to homogenize the range of distances)
    n_smooth is the number of points to set in nice_x.
    """
    n_bottle = min([len(X), len(F)])
    times = X[:n_bottle, 0]
    X = X[:n_bottle, 1:]
    F = F[:n_bottle, 1:]

    nice_x = np.linspace(np.min(X[:, 1:]), np.max(X[:, 1:]), n_smooth)
    dx = nice_x[1] - nice_x[0]
    nice_f = []
    for i in range(len(X[0, :])):
        forces = []
        for x in nice_x:
            closest_ndx = np.argsort(np.abs(X[:, i] - x))[:2]
            if x < np.min(X[:, i]) or x > np.max(X[:, i]):
                forces.append(-1000)
            else:
                forces.append(np.mean(F[closest_ndx, i]))
        nice_f.append(forces)
    nice_f = np.array(nice_f).T
    for i in range(len(nice_x)):
        mask = nice_f[i, :] != -1000
        mean = np.mean(nice_f[i, mask])
        nice_f[i, ~mask] = mean

    nice_g = []
    for f in nice_f.T:
        dg = np.cumsum(f * dx)
        nice_g.append(dg)
    nice_g = np.array(nice_g).T
    nice_g = nice_g[n_ignore:-n_ignore] - nice_g[n_ignore:-n_ignore][-1]
    return (
        nice_x[n_ignore:-n_ignore],
        nice_f[n_ignore:-n_ignore] / 4.184,
        nice_g / 4.184,
    )  # kJ to kcal"""


def protein_similitude_score(data):
    """
    Takes a dataframe with columns aa1 and aa2 and calculates a protein similarity score,
    which reflects how common are aa1 and aa2 in naturally ocurring binding pockets of catecholamines
    """
    max_freq = max(PDB_frequencies.values())
    max_score = 2 * max_freq
    score = [
        0.5 * (PDB_frequencies[aa1] + PDB_frequencies[aa2]) / max_freq
        for aa1, aa2 in data[["aa1", "aa2"]].values
    ]
    return score


class Catecholamine_PMF:
    """
    Class that contains all the information of a catecholamine.
    Each entry describes one nanoreceptor screened

    Catecholamine_PMF.rname -> Residue name of the catecholamine
    Catecholamine_PMF.keys -> Identifier for each nanoreceptor built from aa1 and aa2 in AA_pairs (see Constants.py)
    Catecholamine_PMF.x -> Sampled distances between the analytes and the gold center
    Catecholamine_PMF.f -> Steering force of the analytes at each position
    Catecholamine_PMF.g -> Potential of mean force of the analytes at each position
    Catecholamine_PMF.g_bind -> Binding free energy of each analyte molecule
    Catecholamine_PMF.g_mean -> Binding free energy of the dyad averaged over analyte molecules
    Catecholamine_PMF.g_std -> Standard deviation on the binding free energy of the dyad (computed from Catecholamine_PMF.g_bind)
    Catecholamine_PMF.dangs -> Angle measured for the analyte molecules
    Catecholamine_PMF.data -> Dataframe with aa1, aa2, g_bind, g_mean, and g_std
    Catecholamine_PMF.protein_similitude -> Similarity coefficient reflecting how common are aa1 and aa2 in
    naturally ocurring binding pockets of catecholamines
    """

    def __init__(self, rname, setup):
        self.rname = rname
        self.setup = setup
        pullx = import_pullx(rname, setup=self.setup)
        pullf = import_pullf(rname, setup=self.setup)
        self.keys = ["{}-{}".format(a1, a2) for a1, a2 in AA_pairs]
        self.x, self.f, self.g, self.g_bind, self.g_mean, self.g_std, self.dangs = (
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        for key in self.keys:
            print(key)
            self.x[key], self.f[key], self.g[key] = homogenize_pull(
                pullx[key], pullf[key]
            )
            self.g_bind[key] = reject_outliers(
                [np.min(gibbs) for gibbs in self.g[key].T], iq=1.5
            )
            self.g_mean[key] = boltz_mean(self.g_bind[key])
            self.g_std[key] = boltz_std(self.g_bind[key])
            self.dangs[key] = read_dang(rname, self.x[key], key, setup=self.setup)

        colnames = ["aa1", "aa2", "dgs", "dg_mean", "dg_std"]
        core = [
            [key[:3], key[4:7], g, g_av, g_dev]
            for key, g, g_av, g_dev in zip(
                self.keys,
                self.g_bind.values(),
                self.g_mean.values(),
                self.g_std.values(),
            )
        ]
        self.data = pd.DataFrame(core, columns=colnames)
        self.protein_similitude = protein_similitude_score(self.data)
        self.data["psc"] = self.protein_similitude
