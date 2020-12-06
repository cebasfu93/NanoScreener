import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.4
Z = 18
Nan = 10
N_ignore = 0

LOGACT_THRES=1.0

charged_nps = [1, 2, 3, 16]
anionic_a = [17, 20, 21]
cationic_a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19]

#EXP=Affinity constant, std, ranking
EXP = {#"N8A20": [0, 0],
      #"N14A20": [0, 0],
      #"N9A20" : [0, 0],
      #"N6A21" : [0, 0],
      "N1A16" : [0, 0],
      "N1A17" : [0, 0],
      "N3A10" : [0, 0],
      "N3A11" : [0, 0],
      "N3A18" : [0, 0],
      #"N6A20" : [120, 5],
      "N15A20": [300, 0],
      #"N7A20" : [1100, 0],
      #"N16A20": [4.5*10**4, 0],
      #"N13A20": [5.0*10**4, 0],
      "N2A11" : [8.5*10**4, 1.5*10**4],
      "N2A10" : [9.3*10**4, 1.2*10**4],
      "N1A9"  : [1.2*10**5, 0.2*10**5],
      "N1A8"  : [1.3*10**5, 0.2*10**5],
      "N1A7"  : [2.6*10**5, 0.6*10**5],
      "N1A13" : [2.7*10**5, 0.3*10**5],
      "N1A1"  : [3.6*10**5, 0],
      "N3A19" : [3.6*10**5, 0.1*10**5],
      "N1A15" : [3.7*10**5, 0],
      "N1A12" : [3.9*10**5, 0.3*10**5],
      "N1A10" : [4.1*10**5, 0.4*10**5],
      "N1A14" : [4.6*10**5, 0],
      "N1A11" : [4.8*10**5, 0.5*10**5],
      "N2A18" : [5.1*10**5, 0.3*10**5],
      "N1A18" : [6.1*10**5, 1.4*10**5],
      "N2A19" : [1.3*10**6, 0.1*10**6],
      "N1A19" : [2.2*10**6, 0.1*10**6],
      }

Ns = [int(key.split('A')[0].split("N")[1]) for key in EXP.keys()]
As = [int(key.split('A')[1]) for key in EXP.keys()]
Systems = ["N{}A{}".format(n,a) for n,a in zip(Ns, As)]
act_mean = [EXP["N{}A{}".format(n, a)][0] for n, a in zip(Ns, As)]
act_std = [EXP["N{}A{}".format(n, a)][1] for n, a in zip(Ns, As)]
act_rank = np.argsort(-1*np.array(act_mean))
colors = np.round(cm.rainbow(np.linspace(0, 1, len(Ns))), 2)
colnames = ['Nanoparticle', 'Analyte', 'Act_Mean', 'Act_Std', 'Act_Rank', 'Color']
DATA = pd.DataFrame(zip(Ns, As, act_mean, act_std, act_rank, colors), columns=colnames, index=Systems)
DATA.index.name = "System"

act_means = DATA.Act_Mean.values*1
act_means[act_means==0] = 1.0
act_stds = DATA.Act_Std.values*1
log_means = np.log10(act_means)
DATA['Act_log'] = log_means
err_up = np.log10(act_means+act_stds) - np.log10(act_means)
err_down = np.log10(act_means) - np.log10(act_means-act_stds)
error = np.array([err_down, err_up])
DATA['Act_log_error_up'] = err_up
DATA['Act_log_error_down'] = err_down
Act_Rank_Std_up = []
Act_Rank_Std_down = []

n_act = np.sum(DATA.Act_Mean > 0.0)
for i in range(len(DATA)):
    lower_than = np.sum((DATA.Act_Mean[i] + DATA.Act_Std[i]) >= (DATA.Act_Mean[i:]-DATA.Act_Std[i:]))-1 #-1 to not count itself
    Act_Rank_Std_up.append(lower_than)
    greater_than = np.sum((DATA.Act_Mean[i] - DATA.Act_Std[i]) <= (DATA.Act_Mean[:i]+DATA.Act_Std[:i])) #-1 is absent because of slicing open/closed limits
    Act_Rank_Std_down.append(greater_than)
Act_Rank_Std_up = np.array(Act_Rank_Std_up)
DATA['Act_Rank_Std_up'] = Act_Rank_Std_up*1
DATA['Act_Rank_Std_down'] = Act_Rank_Std_down*1
DATA['Act_Rank_Std_up'].loc[DATA.Act_Mean==0.0] = DATA['Act_Rank_Std_down'].loc[DATA.Act_Mean==0.0]*1
DATA['Act_Rank_Std_down'].loc[DATA.Act_Mean==0.0] = Act_Rank_Std_up[DATA.Act_Mean==0.0]*1
DATA['Active'] = DATA['Act_log'] > LOGACT_THRES
LOGACT_NDX = np.where((DATA['Active'] == False))[0][-1]

DATA = DATA[['Nanoparticle', 'Analyte', 'Act_Mean', 'Act_Std', \
'Act_Rank', 'Act_Rank_Std_up', 'Act_Rank_Std_down', \
'Act_log', 'Act_log_error_up', 'Act_log_error_down', \
'Color', 'Active']]

NP_lonepairs = {1:7, 2:9, 3:9, 5:14, 6:8, 7:8, 8:10, 9:10, 10:8, 11:8, 12:10, 13:8, 14:10, 15:8, 16:0}
NP_hydrogens = {1:0, 2:1, 3:1, 5:1, 6:1, 7:2, 6:1, 7:2, 8:1, 9:2, 10:0, 11:0, 12:2, 13:3, 14:1, 15:1, 16:3}
AN_lonepairs = {1:0, 6:0, 7:0, 8:2, 9:4, 10:2, 11:2, 12:4, 13:2, 14:5, 15:2, 16:5, 17:7, 18:0, 19:0, 20:7, 21:6}
AN_hydrogens = {1:3, 6:2, 7:4, 8:4, 9:5, 10:3, 11:3, 12:3, 13:5, 14:3, 15:3, 16:3, 17:1, 18:3, 19:3, 20:1, 21:1} #Fluors were included
NP_pcharges = {1:0, 2:0, 3:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:1}
NP_ncharges = {1:1, 2:1, 3:1, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0}
A_pcharges = {1:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:0, 18:1, 19:1, 20:0, 21:0}
A_ncharges = {1:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:1, 17:1, 18:0, 19:0, 20:1, 21:1}
NP_arrings = {1:0, 2:1, 3:1, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0}
A_arrings = {1:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:2, 14:1, 15:1, 16:1, 17:1, 18:2, 19:4, 20:1, 21:1}

np_lonepairs = [NP_lonepairs[n] for n in Ns]
np_hydrogens = [NP_hydrogens[n] for n in Ns]
a_lonepairs = [AN_lonepairs[a] for a in As]
a_hydrogens = [AN_hydrogens[a] for a in As]
np_pcharges = [NP_pcharges[n] for n in Ns]
np_ncharges = [NP_ncharges[n] for n in Ns]
a_pcharges = [A_pcharges[a] for a in As]
a_ncharges = [A_ncharges[a] for a in As]
np_arrings = [NP_arrings[n] for n in Ns]
a_arrings = [A_arrings[a] for a in As]


if __name__ == '__main__':
    print('This statement will be executed only if this script is called directly')
