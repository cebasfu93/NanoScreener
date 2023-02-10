import numpy as np
from scipy import optimize
from DEPENDENCIES.constants import *


def read_text_file(fname):
    """
    Reads text file with path fname
    Ignores lines with "#" and "@"
    Return array
    """
    f = open(fname, "r")
    fl = f.readlines()
    f.close()

    data = []
    for line in fl:
        if "#" not in line and "@" not in line:
            data.append(line.split())
    clean = np.array(data, dtype='float')
    return clean


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
    return np.exp(-0.5 * np.power((x - mu) / sig, 2.)) / (sig * (2 * np.pi)**0.5)


def reject_outliers(data, iq=1.5):
    """
    Slices the 1D array data keeping only the values within iq times the interquartile distance from the median.
    """
    IQ = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    mask = np.logical_and(data < np.median(data) + iq * IQ, data > np.median(data) - iq * IQ)
    return data[mask]


def boltz_mean(x, T=300):
    """
    Calculates Boltzmann-weighted average of the variable x at a temperature T
    """
    RT = 0.001987 * T  # kcal/mol
    nom = np.sum(np.multiply(np.exp(x / RT), x))
    denom = np.sum(np.exp(x / RT))
    return nom / denom


def boltz_std(x, T=300):
    """
    Calculates Boltzmann-weighted standard deviation (i.e. square root of the second statistical moment) of the variable x at a temperature T
    """
    RT = 0.001987 * 300  # kcal/mol

    mean2 = np.sum(np.multiply(np.exp(x / RT), x**2)) / np.sum(np.exp(x / RT))
    mean = np.sum(np.multiply(np.exp(x / RT), x)) / np.sum(np.exp(x / RT))
    return (mean2 - mean**2)**0.5


def import_MD_data(S, dist_suffix, data=DATA, stat_tini=3, stat_tfin=4):
    dists = np.array([[read_text_file("./S{}/DISTS/{}S{}_{}_dcom.sfu".format(s,
                                                                             NA, s, dist_suffix))[:, 1:] for NA in data.index] for s in S])
    rogs = np.array([[read_text_file("./S{}/ROGS/{}S{}_gyr.xvg".format(s, NA, s))[:, 1]
                      for NA in data.index] for s in S])
    sasas = np.array([[read_text_file("./S{}/SASA/{}S{}_sasa.xvg".format(s, NA, s))[:, 1]
                       for NA in data.index] for s in S])
    times = read_text_file(
        "./S{}/DISTS/{}S{}_{}_dcom.sfu".format(S[0], data.index[0], S[0], dist_suffix))[:, 0]
    print(dists.shape)

    dists_all_runs = []
    for nn in range(dists.shape[1]):
        to_catenate = [d for d in dists[:, nn, :, :]]
        dists_all_runs.append(np.concatenate(to_catenate, axis=1))
    data['Distances'] = dists_all_runs

    t_mask = np.logical_and(times > stat_tini, times < stat_tfin)
    rogs_warm = rogs[:, :, t_mask]
    sasas_warm = sasas[:, :, t_mask]
    data['Rogs'] = list(np.mean(rogs, axis=0))
    data['Rog_Mean'] = np.mean(rogs_warm, axis=(0, 2))
    data['Rog_Std'] = np.std(rogs_warm, axis=(0, 2))
    data['Sasas'] = list(np.mean(sasas, axis=0))
    data['Sasa_Mean'] = np.mean(sasas_warm, axis=(0, 2))
    data['Sasa_Std'] = np.std(sasas_warm, axis=(0, 2))
    return times, data


def confusion_matrix_sfu(active, pred_active):
    """
    active and pred_active are two 1D boolean arrays with
    True if the entry (dyad) is known or predicted to be active
    False if the entry (dyad) is known or predicted to be inactive

    Returns number of
    true positives (TP),
    true negatives (TN),
    false positives (FP), and
    false negatives (FN)
    """
    TP = np.sum(pred_active + active == 2)
    TN = np.sum(pred_active + active == 0)
    FP = np.sum(pred_active - active == 1)
    FN = np.sum(pred_active - active == -1)
    return TP, TN, FP, FN


def evaluate_metric(metric_name, best_i, metric_params):
    if metric_name == "Bye_time":
        metric = calc_bye_time
    elif metric_name == "Bound_fraction":
        metric = calc_bound_fraction
    elif metric_name == "Weighted_fraction":
        metric = calc_weighted_bound_fraction
    else:
        raise Exception("Valid names: 'Bye_time', 'Bound_fraction', 'Weighted_fraction'")
    res = metric(best_i, *metric_params)
    return list(res)


def true_positives(best_j, data=DATA):
    """
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    best_j is the best (chosen) free energy threshold

    Returns Dataframe with the true positive dyads
    """
    tp = data.loc[data.Active.astype(int) + (data.Score.apply(np.mean) > best_j).astype(int) == 2]
    return tp


def false_positives(best_j, data=DATA):
    """
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    best_j is the best (chosen) free energy threshold

    Returns Dataframe with the false positive dyads
    """
    fp = data.loc[data.Active.astype(int) - (data.Score.apply(np.mean) > best_j).astype(int) == -1]
    return fp


def false_negatives(best_j, data=DATA):
    """
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    best_j is the best (chosen) free energy threshold

    Returns Dataframe with the false negatives dyads
    """
    fn = data.loc[data.Active.astype(int) - (data.Score.apply(np.mean) > best_j).astype(int) == 1]
    return fn


def true_negatives(best_j, data=DATA):
    """
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    best_j is the best (chosen) free energy threshold

    Returns Dataframe with the true negatives dyads
    """
    tn = data.loc[data.Active.astype(int) + (data.Score.apply(np.mean) > best_j).astype(int) == 0]
    return tn


if __name__ == '__main__':
    print('This statement will be executed only if this script is called directly')
