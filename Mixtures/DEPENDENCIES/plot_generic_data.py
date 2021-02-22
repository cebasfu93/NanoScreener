from DEPENDENCIES.constants import *
from DEPENDENCIES.processing import *
import numpy as np
import matplotlib.pyplot as plt


# DEPRECATED
def plot_experimental_logK():
    subplot_kw = {'xlim': (-0.5, max(DATA.Act_Rank) + 0.5),
                  'xticks': np.linspace(0, max(DATA.Act_Rank), max(DATA.Act_Rank) + 1)}
    fig, axs = plt.subplots(figsize=(14, 8), sharex=True, ncols=1, nrows=2,
                            subplot_kw=subplot_kw, gridspec_kw={'hspace': 0.12, 'wspace': 0.0})
    plt.xlabel("Log(Binding constant)", fontsize=Z)
    for ax in axs:
        ax.grid()
        ax.set_ylabel("Log(Activity)", fontsize=Z)
        ax.tick_params(labelsize=Z)
        ax.errorbar(DATA.Act_Rank, DATA.Act_log, yerr=error, fmt='o', capsize=16, markeredgecolor='k',
                    markeredgewidth=1, markerfacecolor='b', markersize=8, linewidth=2, color='k')
    axs[0].set_ylim(4.4, 6.4)
    axs[0].spines['bottom'].set_visible(False)
    axs[1].set_ylim(1.6, 3.1)
    axs[1].spines['top'].set_visible(False)
    axs[-1].set_xlabel("Exp. Ranking", fontsize=Z)
    plt.show()
    plt.close()


def plot_all_rog(times, data=DATA, nw=50, kwargs={}):
    nrow = len(data.Nanoparticle.value_counts())
    ncol = max(data.Nanoparticle.value_counts())

    fig, axs = plt.subplots(figsize=(2 * ncol, 2 * nrow), ncols=ncol, nrows=nrow, sharex=True, sharey=True,
                            subplot_kw={'ylim': kwargs.get('ylim', (0.9, 1.3)),
                                        'xlim': kwargs.get('xlim', (0, 10)), 'xticks': kwargs.get('xticks', [0, 5])},
                            gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    for n, nn in enumerate(np.sort(data.Nanoparticle.unique())[::-1]):
        axs[n, 0].set_ylabel("NP{}\n R.O.G.\n(nm)".format(nn), fontsize=Z)
        axs[n, 0].set_yticks(kwargs.get('yticks', [0.9, 1.1]))
        for (idx, row), ax in zip(data[data.Nanoparticle == nn].iterrows(), axs[n]):
            ax.grid()
            ax.tick_params(labelsize=Z)
            ax.plot(times, row.Rogs, color=row.Color, lw=1, alpha=0.4)
            ax.plot(times[nw:], ma(row.Rogs, nw), color=row.Color, lw=2)
    for ax in axs[-1, :]:
        ax.set_xlabel("Time (ns)", fontsize=Z)
    plt.show()
    plt.close()


def plot_rog_byact(data=DATA):
    fig, axs = plt.subplots(figsize=(15, 5))
    axs.tick_params(labelsize=Z - 4)
    axs.set_xticklabels(data.index, rotation=90)
    axs.grid()
    axs.set_ylim(0.9, 1.3)
    axs.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
    axs.set_xlabel("System", fontsize=Z)
    axs.set_ylabel("R.O.G. (nm)", fontsize=Z)
    axs.bar(data.index, data['Rog_Mean'], yerr=data['Rog_Std'],
            color=data.Color, linewidth=2, ec='k', width=0.7, capsize=3)
    plt.show()
    plt.close()


def plot_rog_bynp(data=DATA):
    df_sorted = data.sort_values(['Nanoparticle', 'Act_Rank'])
    fig, axs = plt.subplots(figsize=(15, 5))
    axs.tick_params(labelsize=Z - 4)
    axs.set_xticklabels(df_sorted['Nanoparticle'])
    axs.grid()
    axs.set_ylim(0.9, 1.3)
    axs.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
    axs.set_xlabel("Nanoparticle", fontsize=Z)
    axs.set_ylabel("R.O.G. (nm)", fontsize=Z)
    axs.bar(df_sorted.index, df_sorted['Rog_Mean'], yerr=df_sorted['Rog_Std'],
            color=df_sorted.Color, linewidth=2, ec='k', width=0.7, capsize=3)
    plt.show()
    plt.close()


def plot_all_sasa(times, data=DATA, nw=50, kwargs={}):
    nrow = len(data.Nanoparticle.value_counts())
    ncol = max(data.Nanoparticle.value_counts())

    fig, axs = plt.subplots(figsize=(2 * ncol, 2 * nrow), ncols=ncol, nrows=nrow, sharex=True, sharey=True, subplot_kw={'ylim': kwargs.get(
        'ylim', (80, 160)), 'xlim': kwargs.get('xlim', (0, 10)), 'xticks': kwargs.get('xticks', [0, 5])}, gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    for n, nn in enumerate(np.sort(data.Nanoparticle.unique())[::-1]):
        axs[n, 0].set_ylabel("NP{}\nS.A.S.A.\n".format(nn) + r"$nm^2$", fontsize=Z)
        axs[n, 0].set_yticks(kwargs.get('yticks', [80, 120]))
        for (idx, row), ax in zip(data[data.Nanoparticle == nn].iterrows(), axs[n]):
            ax.grid()
            ax.tick_params(labelsize=Z)
            ax.plot(times, row.Sasas, color=row.Color, lw=1, alpha=0.4)
            ax.plot(times[nw:], ma(row.Sasas, nw), color=row.Color, lw=2)
    for ax in axs[-1, :]:
        ax.set_xlabel("Time (ns)", fontsize=Z)
    plt.show()
    plt.close()


def plot_sasa_byact(data=DATA):
    fig, axs = plt.subplots(figsize=(15, 5))
    axs.tick_params(labelsize=Z - 4)
    axs.set_xticklabels(data.index, rotation=90)
    axs.grid()
    axs.set_ylim(80, 160)
    axs.set_yticks([80, 100, 120, 140, 160])
    axs.set_xlabel("System", fontsize=Z)
    axs.set_ylabel("S.A.S.A. ($nm^2$)", fontsize=Z)
    axs.bar(data.index, data['Sasa_Mean'], yerr=data['Sasa_Std'],
            color=data.Color, linewidth=2, ec='k', width=0.7, capsize=3)
    plt.show()
    plt.close()


def plot_sasa_bynp(data=DATA):
    df_sorted = data.sort_values(['Nanoparticle', 'Act_Rank'])
    fig, axs = plt.subplots(figsize=(15, 5))
    axs.tick_params(labelsize=Z - 4)
    axs.set_xticklabels(df_sorted['Nanoparticle'])
    axs.grid()
    axs.set_ylim(80, 160)
    axs.set_yticks([80, 100, 120, 140, 160])
    axs.set_xlabel("Nanoparticle", fontsize=Z)
    axs.set_ylabel("S.A.S.A. ($nm^2$)", fontsize=Z)
    axs.bar(df_sorted.index, df_sorted['Sasa_Mean'], yerr=df_sorted['Sasa_Std'],
            color=df_sorted.Color, linewidth=2, ec='k', width=0.7, capsize=3)
    plt.show()
    plt.close()


if __name__ == '__main__':
    print('This statement will be executed only if this script is called directly')
