import numpy as np
from scipy import optimize
from DEPENDENCIES.constants import *

def read_text_file(fname):
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

def reject_outliers(data, iq=1.5):
    IQ = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    mask = np.logical_and(data<np.median(data)+iq*IQ, data>np.median(data)-iq*IQ)
    return data[mask]

def boltz_mean(x):
    RT = 0.001987 * 300 #kcal/mol
    nom = np.sum(np.multiply(np.exp(x/RT), x))
    denom = np.sum(np.exp(x/RT))
    return nom/denom

def boltz_std(x):
    RT = 0.001987 * 300 #kcal/mol
    
    mean2 = np.sum(np.multiply(np.exp(x/RT), x**2))/np.sum(np.exp(x/RT))
    mean = np.sum(np.multiply(np.exp(x/RT), x))/np.sum(np.exp(x/RT))
    #return (np.sum(np.multiply(np.exp(x/RT), (x-mean)**2))/np.sum(np.exp(x/RT)))**0.5 #two returns equivalent
    return (mean2 - mean**2)**0.5

def import_MD_data(S, dist_suffix, data=DATA, stat_tini=3, stat_tfin=4):
    dists = np.array([[read_text_file("./S{}/DISTS/{}S{}_{}_dcom.sfu".format(s, NA, s, dist_suffix))[:,1:] for NA in data.index] for s in S])
    rogs = np.array([[read_text_file("./S{}/ROGS/{}S{}_gyr.xvg".format(s, NA, s))[:,1] for NA in data.index] for s in S])
    sasas = np.array([[read_text_file("./S{}/SASA/{}S{}_sasa.xvg".format(s, NA, s))[:,1] for NA in data.index] for s in S])
    times = read_text_file("./S{}/DISTS/{}S{}_{}_dcom.sfu".format(S[0], data.index[0], S[0], dist_suffix))[:,0]
    print(dists.shape)

    dists_all_runs = []
    for nn in range(dists.shape[1]):
        to_catenate = [d for d in dists[:,nn,:,:]]
        dists_all_runs.append(np.concatenate(to_catenate, axis=1))
    data['Distances'] = dists_all_runs

    t_mask = np.logical_and(times>stat_tini, times<stat_tfin)
    rogs_warm = rogs[:,:,t_mask]
    sasas_warm = sasas[:,:,t_mask]
    data['Rogs'] = list(np.mean(rogs, axis=0))
    data['Rog_Mean'] = np.mean(rogs_warm, axis=(0,2))
    data['Rog_Std'] = np.std(rogs_warm, axis=(0,2))
    data['Sasas'] = list(np.mean(sasas, axis=0))
    data['Sasa_Mean'] = np.mean(sasas_warm, axis=(0,2))
    data['Sasa_Std'] = np.std(sasas_warm, axis=(0,2))
    return times, data

def confusion_matrix_sfu(active, pred_active):
    TP = np.sum(pred_active+active==2)
    TN = np.sum(pred_active+active==0)
    FP = np.sum(pred_active-active==1)
    FN = np.sum(pred_active-active==-1)
    return TP, TN, FP, FN

def calc_bye_time(eta, times, data=DATA):
    """Determines the first moment in which each analyte departs eta*rog from the gold COM.
    The N_ignore analytes that unbind the fastest are discarded"""
    d_shape = data.iloc[0].Distances.shape
    res = np.ones((len(data), d_shape[1]))
    for i in range(len(data)):
        rog = data['Rog_Mean'][i]
        for j in range(d_shape[1]):
            try:
                when = np.where(data.Distances[i][:,j] >= eta*rog)[0][0]
            except:
                when = -1
            res[i,j] = times[when]
    res = np.sort(res, axis=1)[:,N_ignore:]
    return res

def calc_bound_fraction(eta, times, data=DATA):
    """Calculates fraction, normalized to frames and analytes, of time that the analytes are
    under eta*rog from the gold COM"""
    d_shape = data.iloc[0].Distances.shape
    res = np.zeros((len(data), d_shape[1]))
    for i in range(len(data)):
        rog = data['Rog_Mean'][i]
        res[i,:] = np.sum(data.Distances[i]<=eta*rog, axis=0)/(d_shape[0])
    return res

def weighting_factor(t, th, endtime=6):
    """Calculates the exponential function so that the integral under th is 0.2 and the integral until endtime is 1"""
    beta_func = lambda x: (np.exp(th*x) - 0.2*np.exp(x*endtime) - 0.8)**2
    beta = optimize.minimize(beta_func, x0=1.5).x[0]
    alpha = beta/(np.exp(beta*endtime)-1)
    weight = alpha*np.exp(beta*t)
    return weight

def calc_weighted_bound_fraction(eta, theta, times, endtime, data=DATA):
    """Weights the binding events with the exponential function from weighting_factor"""
    dt = times[-1]/len(times)
    weights = weighting_factor(times, theta, endtime=endtime)
    n_ana = len(data.Distances[0][0])
    res = np.zeros((len(data),n_ana))
    for i in range(len(data)):
        rog = data['Rog_Mean'][i]
        bt = np.sum(data.Distances[i]<=eta*rog, axis=1)/n_ana
        btw = np.sum(bt*weights)*dt
        btw_var = np.sum((bt-btw)**2*weights)*dt
        res[i] = np.random.normal(loc=btw, scale=btw_var, size=n_ana)
    return res

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

def metric_gridsearch(i_space, j_space, metric_name, data=DATA, metric_params=()):
    """Calculates matrices with acuracy, sensitivity and specificity for a given metric.
    metric = 'bye_time'/'bound_fraction','weighted_binding'"""
    if metric_name == "Bye_time":
        metric = calc_bye_time
    elif metric_name == "Bound_fraction":
        metric = calc_bound_fraction
    elif metric_name == "Weighted_fraction":
        metric = calc_weighted_bound_fraction
    else:
        raise Exception("Valid names: 'Bye_time', 'Bound_fraction', 'Weighted_fraction'")
    accuracy, sensitivity, specificity = [], [], []
    for i in i_space:
        met_val = metric(i, *metric_params)
        acc, sens, spec = [], [], []
        for j in j_space:
            if metric_name == "Weighted_binding":
                pred_active = (met_val > j).astype('float')
            else:
                pred_active = (np.mean(met_val, axis=1) > j).astype('float')
            active = data.Active.astype('float')
            tp, tn, fp, fn = confusion_matrix_sfu(pred_active, active)
            acc.append((tp+tn)/(tp+tn+fp+fn))
            sens.append(tp/(tp+fn))
            spec.append(tn/(tn+fp))
        accuracy.append(acc)
        sensitivity.append(sens)
        specificity.append(spec)
    accuracy, sensitivity, specificity = np.array(accuracy), np.array(sensitivity), np.array(specificity)
    return accuracy, sensitivity, specificity

def choose_best_param_ndx(accuracy, sensitivity, specificity, opt_criter='accuracy', decide_criter='sensspec', blo=-1, blp=0):
    """Determines the best combination of i and j parameters. The best performance is chosen based on opt_criter, but,
    for identical performances, the best is chosen based on decide_criter.
    blo selects the index of opt_criter considered the best, and blp selects the index of decide_criter considered the best."""
    if opt_criter == 'accuracy':
        opt_array = accuracy
    elif opt_criter == 'sensspec':
        opt_array = 0.5*(sensitivity+specificity)
    else:
        raise Exception("Wrong optimizing criterium")
    if decide_criter == 'accuracy':
        decide_array = accuracy
    elif decide_criter == 'sensspec':
        decide_array = 0.5*(sensitivity+specificity)
    else:
        raise Exception("Wrong decision criterium heny")

    ndxs_i, ndxs_j = np.where(opt_array==np.sort(opt_array.flatten())[blo])
    decide_max = np.max(decide_array[ndxs_i, ndxs_j])
    top_ndxs = np.where(decide_array==decide_max)
    best_i_ndx = [i for i, j in zip(ndxs_i, ndxs_j) if i in top_ndxs[0] and j in top_ndxs[1]]
    best_j_ndx = [j for i, j in zip(ndxs_i, ndxs_j) if i in top_ndxs[0] and j in top_ndxs[1]]
    print("Equivalent (i,j) solutions: ", list(zip(best_i_ndx, best_j_ndx)))
    best_i_ndx = best_i_ndx[blp]
    best_j_ndx = best_j_ndx[blp]
    print("Chosen indeces: ({},{})".format(best_i_ndx, best_j_ndx))
    print("Accuracy at best point: {:.2f}".format(accuracy[best_i_ndx, best_j_ndx]))
    print("Sensitivity at best point: {:.2f}".format(sensitivity[best_i_ndx, best_j_ndx]))
    print("Specificity at best point: {:.2f}".format(specificity[best_i_ndx, best_j_ndx]))
    return best_i_ndx, best_j_ndx

def true_positives(best_j, data=DATA):
    tp = data.loc[data.Active.astype(int) + (data.Score.apply(np.mean) > best_j).astype(int)==2]
    return tp

def false_positives(best_j, data=DATA):
    fp = data.loc[data.Active.astype(int) - (data.Score.apply(np.mean) > best_j).astype(int)==-1]
    return fp

def false_negatives(best_j, data=DATA):
    fn = data.loc[data.Active.astype(int) - (data.Score.apply(np.mean) > best_j).astype(int)==1]
    return fn

def true_negatives(best_j, data=DATA):
    tn = data.loc[data.Active.astype(int) + (data.Score.apply(np.mean) > best_j).astype(int)==0]
    return tn

#fs = np.linspace(0,1, len(times))

if __name__ == '__main__':
    print('This statement will be executed only if this script is called directly')
