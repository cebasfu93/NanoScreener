from Constants import *
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import gaussian, ma, boltz_mean, boltz_std
from matplotlib import cm


def plot_all_angs(cath):
    """
    cath is a Catecholamine_PMF object (see Preprocessing.py)

    Plots the columns in cath.dangle
    Usually this is the orientation of the analyte vs its distance to the gold
    The result is a grid of plots, one plot for each system

    The angle of each analyte is plotted in transparency and the mean profile in opaque
    """
    fig, axs = plt.subplots(figsize=(28, 28), ncols=len(AA_ordered), nrows=len(AA_ordered), sharex=True, sharey=True, subplot_kw={
                            'xlim': (0, 5), 'ylim': (0, 180), 'xticks': [0, 2.5], 'yticks': [0, 60, 120]}, gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    for a1, aa1 in enumerate(AA_ordered):
        axs[a1, 0].set_ylabel("{}\nAngle\ndeg".format(aa1), fontsize=Z - 4)
        axs[0, a1].set_title(aa1, fontsize=Z)
        axs[-1, a1].set_xlabel("Distance\n(nm)", fontsize=Z - 4)
        for a2, aa2 in enumerate(AA_ordered):
            key = "{}-{}".format(aa1, aa2).upper()
            x = cath.x.get(key, [])
            angle = cath.dangs.get(key, [])
            if x != []:
                for ang in angle.T:
                    axs[a1, a2].plot(x, ang, lw=1, c=(0.9, 0.5, 0), alpha=0.05)
                mean_ang = np.mean(angle, axis=1)
                axs[a1, a2].plot(x, mean_ang, lw=1, c=(0.9, 0.5, 0), alpha=1)
    for ax in axs.flatten():
        ax.tick_params(labelsize=Z)
    plt.savefig("Plots/{}/all_angles.png".format(cath.rname), format='png')
    plt.show()
    plt.close()


def plot_all_forces(cath):
    """
    cath is a Catecholamine_PMF object (see Preprocessing.py)

    Plots the cath.f vs cath.x for all the systems
    Usually this is the steering force vs the position of the analyte from the gold center
    The result is a grid of plots, one plot for each system

    The force of each analyte is plotted in transparency and the mean profile in opaque
    """
    fig, axs = plt.subplots(figsize=(28, 28), ncols=len(AA_ordered), nrows=len(AA_ordered), sharex=True, sharey=True, subplot_kw={
                            'xlim': (0, 5), 'ylim': (-75, 75), 'xticks': [0, 2.5], 'yticks': [-75, -25, 25]},
                            gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    for a1, aa1 in enumerate(AA_ordered):
        axs[a1, 0].set_ylabel("{}\nForce\nkcal\n".format(aa1) + r"$mol^{-1}nm^{-1}$", fontsize=Z - 4)
        axs[0, a1].set_title(aa1, fontsize=Z)
        axs[-1, a1].set_xlabel("Distance\n(nm)", fontsize=Z - 4)
        for a2, aa2 in enumerate(AA_ordered):
            key = "{}-{}".format(aa1, aa2).upper()
            x = cath.x.get(key, [])
            f = cath.f.get(key, [])
            if x != []:
                for force in f.T:
                    axs[a1, a2].plot(x, force, lw=1, c=(0.5, 0, 0.4), alpha=0.05)
                mean_f = np.mean(f, axis=1)
                axs[a1, a2].plot(x, mean_f, lw=1, c=(0.5, 0, 0.4), alpha=1)
    for ax in axs.flatten():
        ax.tick_params(labelsize=Z)
    plt.savefig("Plots/{}/all_forces.png".format(cath.rname), format='png')
    plt.show()
    plt.close()


def plot_all_dgs(cath):
    """
    cath is a Catecholamine_PMF object (see Preprocessing.py)

    Plots the cath.x vs cath.g for all the systems
    Usually this is the potential of mean force vs the position of the analyte from the gold center
    The result is a grid of plots, one plot for each system

    The PMF profile of each analyte is plotted in transparency and the mean profile in opaque
    """
    fig, axs = plt.subplots(figsize=(28, 28), ncols=len(AA_ordered), nrows=len(AA_ordered), sharex=True, sharey=True, subplot_kw={
                            'xlim': (0, 5), 'ylim': (-30, 15), 'xticks': [0, 2.5], 'yticks': [-30, -15, 0]},
                            gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    for a1, aa1 in enumerate(AA_ordered):
        axs[a1, 0].set_ylabel("{}\nFree energy\nkcal ".format(aa1) + r"$mol^{-1}$", fontsize=Z - 4)
        axs[0, a1].set_title(aa1, fontsize=Z)
        axs[-1, a1].set_xlabel("Distance\n(nm)", fontsize=Z - 4)
        for a2, aa2 in enumerate(AA_ordered):
            key = "{}-{}".format(aa1, aa2).upper()
            x = cath.x.get(key, [])
            g = cath.g.get(key, [])
            if x != []:
                axs[a1, a2].axhline(0, c='k', lw=1)
                for gibbs in g.T:
                    axs[a1, a2].plot(x, gibbs, lw=2, c=(0, 0.6, 1), alpha=0.1)
                mean_g = np.mean(g, axis=1)
                axs[a1, a2].plot(x, mean_g, lw=2.5, c=(0, 0.6, 1), alpha=1)
    for ax in axs.flatten():
        ax.tick_params(labelsize=Z)
    plt.savefig("Plots/{}/all_gibbs.png".format(cath.rname), format='png')
    plt.show()
    plt.close()


def plot_dg_distributions(cath):
    """
    cath is a Catecholamine_PMF object (see Preprocessing.py)

    Plots a histogram with data from cath.g
    Usually this is histograms the free energy differences obtained from each analyte
    The result is a grid of plots, one plot for each system

    The PMF profile of each analyte is plotted in transparency and the mean profile in opaque
    """
    g_space = np.linspace(-30, 15, 500)
    fig, axs = plt.subplots(figsize=(28, 28), ncols=len(AA_ordered), nrows=len(AA_ordered), sharex=True, sharey=True, subplot_kw={
                            'xlim': (-30, 15), 'xticks': [-30, -15, 0], 'ylim': (0, 3.0), 'yticks': [0, 0.5, 1.0, 1.5, 2.0, 2.5]},
                            gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    for a1, aa1 in enumerate(AA_ordered):
        axs[a1, 0].set_ylabel("{}\nProb.\ndensity".format(aa1), fontsize=Z)
        axs[0, a1].set_title(aa1, fontsize=Z)
        axs[-1, a1].set_xlabel("Free energy\nkcal" + r"$  mol^{-1}$", fontsize=Z)
        for a2, aa2 in enumerate(AA_ordered):
            key = "{}-{}".format(aa1, aa2).upper()
            x = cath.x.get(key, [])
            g = cath.g.get(key, [])
            if x != []:
                g_mean, g_std = cath.g_mean[key], cath.g_std[key]
                gauss = gaussian(g_space, g_mean, g_std)
                axs[a1, a2].plot(g_space, gauss, lw=3, c=(0, 0.6, 1))
                # axs[a1,a2].axvline(g_mean, ymin=0, ymax=np.max(gauss)/0.15, color=(0,0.3,0.5), lw=2, ls='-')
    for ax in axs.flatten():
        ax.tick_params(labelsize=Z)
    plt.savefig("Plots/{}/all_dg_distr.png".format(cath.rname), format='png')
    plt.show()
    plt.close()


def plot_dg_map(cath, show_mean=True, show_std=False, svg=False, THRESH=0, dg_min=3, dg_max=18):
    """
    cath is a Catecholamine_PMF object (see Preprocessing.py)

    Plots a heat map with the color representing the free energy of the dyad (averaged over analyte molecules)

    If show_mean, the mean numeric value is written in the cells with binding energy lower than THRESH
    If show_mean, the standard error numeric value is written in the cells with binding energy lower than THRESH
    If svg, the plot is saved in .svg format
    dg_min and dg_max set the lower and upper limits of the colormap
    """
    cmap = 'plasma'
    pic = np.zeros((len(AA_ordered), len(AA_ordered)))
    pic_downlim = np.zeros_like(pic)
    for a1, aa1 in enumerate(AA_ordered):
        for a2, aa2 in enumerate(AA_ordered):
            key = "{}-{}".format(aa1, aa2).upper()
            pic[a1, a2] = -1 * cath.g_mean.get(key, np.nan)
            # if the mean - std is lower than THRESH, then it should also be visible. This array does that
            pic_downlim[a1, a2] = -1 * (cath.g_mean.get(key, np.nan) - cath.g_std.get(key, np.nan))
    fig, ax = plt.subplots(figsize=(9, 9), ncols=1, nrows=1)
    cax = ax.imshow(pic, cmap=cmap, vmin=dg_min, vmax=dg_max)
    # This is just an overlapping plot that will make more opaque the cells corresponding to weak binders
    pic2 = pic_downlim.copy()
    pic2[pic2 < THRESH - 1.0] = 0  # 1.0 is expected deviation from PMF methods
    pic2[pic2 >= THRESH - 1.0] = 20  # 1.0 is expected deviation from PMF methods
    ax.imshow(pic2, cmap='binary_r', alpha=0.3)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.linspace(0, len(AA_ordered) - 1, len(AA_ordered)))
    ax.set_xticks(np.linspace(-0.5, len(AA_ordered) - 0.5, len(AA_ordered) + 1), minor=True)
    ax.set_xticklabels(AA_ordered)
    ax.set_yticks(np.linspace(0, len(AA_ordered) - 1, len(AA_ordered)))
    ax.set_yticks(np.linspace(-0.5, len(AA_ordered) - 0.5, len(AA_ordered) + 1), minor=True)
    ax.set_yticklabels(AA_ordered)
    ax.tick_params(labelsize=Z, size=0, width=1.6)
    ax.set_ylabel('AA1', fontsize=Z + 4)
    ax.set_xlabel('AA2', fontsize=Z + 4)
    ax.xaxis.grid(which='minor', lw=2, c='k')
    ax.yaxis.grid(which='minor', lw=2, c='k')

    if show_mean:
        for a1, aa1 in enumerate(AA_ordered):
            for a2, aa2 in enumerate(AA_ordered):
                if pic[a1, a2] > THRESH - 1.0:  # THRESHOLD. 1.0 is expected deviation from PMF methods
                    label = "{:.1f}".format(pic[a1, a2])
                    if show_std:
                        label += "\n{:.1f}".format(cath.g_std["{}-{}".format(aa1, aa2).upper()])
                    ax.text(a2, a1, label, color='k', ha='center',
                            va='center', fontsize=Z - 1)  # , weight='bold'

    a = plt.axes([0.95, 0.125, 0.04, 0.76])
    cbar = fig.colorbar(cax, ax=a, cax=a, ticks=[3, 6, 9, 12, 13.6, 15, 18])
    cbar.ax.tick_params(labelsize=Z + 4, size=6, width=1.6)
    cbar.ax.set_ylabel(r"-Binding free energy (kcal $mol^{-1}$)", fontsize=Z + 4)
    cbar.ax.set_ylim(dg_min, dg_max)

    if svg:
        plt.savefig("Plots/{}/all_dg_map.png".format(cath.rname),
                    format='png', bbox_inches='tight', dpi=300)
        plt.savefig("Plots/{}/all_dg_map.svg".format(cath.rname), format='svg', bbox_inches='tight')
    else:
        plt.savefig("Plots/{}/all_dg_map.png".format(cath.rname),
                    format='png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_aa_dependence(cath):
    """
    cath is a Catecholamine_PMF object (see Preprocessing.py)

    Groups data according to cath.aa1 (the innermost amino acid) and makes boxplot for each possible amino acid
    Does the same grouping by cath.aa2 (the second amino acid)

    The result is one figure with two plots, one grouping by cath.aa1 the other by cath.aa2
    """
    Z = 16
    colors = cm.terrain(np.linspace(0, 0.85, len(AA_ordered)))
    faded_colors = [(*c[:3], 0.8) for c in colors]
    faded_colors = [(*c[:3], 0.4) for c in colors]
    bp = dict(linestyle='-', lw=1.5, color='k', facecolor='r')
    fp = dict(marker='o', ms=8, ls='none', mec='k', mew=2)
    mp = dict(ls='-', lw=1., color='k')
    cp = dict(ls='-', lw=1.5, color='k')
    wp = dict(ls='-', lw=1.5, color='k')

    fig, axs = plt.subplots(figsize=(7, 5), ncols=1, nrows=2, gridspec_kw={'hspace': 0.4})
    plt.ylabel("Free energy (kcal" + r"$mol^{-1}$)", fontsize=Z)
    grouping_col = ['aa1', 'aa2']
    xlabels = ['AA1', 'AA2']

    xx = np.linspace(1, len(AA_ordered), len(AA_ordered))
    for ax, g_col, xlab in zip(axs, grouping_col, xlabels):
        ax.tick_params(labelsize=Z, size=6, width=1.6)
        tmp_df = cath.data.copy()
        tmp_df.dgs = tmp_df.dgs.apply(list)
        g_pts = tmp_df.groupby(g_col).agg({'dgs': 'sum'}).dgs.loc[[aa.upper() for aa in AA_ordered]]
        for g_aa, i, c, fc in zip(g_pts, xx, colors, faded_colors):
            bpl = ax.boxplot(g_aa, positions=[i], widths=0.6, patch_artist=True,
                             boxprops=bp, flierprops=fp, medianprops=mp, capprops=cp, whiskerprops=wp)
            bpl['boxes'][0].set_facecolor(c)
            bpl['fliers'][0].set_markerfacecolor(fc)
        ax.set_ylim(-30, 0)
        ax.set_xlim(0.5, len(AA_ordered) + 0.5)
        ax.set_xticks(xx)
        ax.set_xticklabels(AA_ordered, fontsize=Z)
        ax.set_yticks([-30, -20, -10, 0])
        ax.set_xlabel(xlab, fontsize=Z)
        # ax.set_ylabel("Free energy\n(kcal"+r"$mol^{-1}$)", fontsize=Z)
        ax.grid()
    plt.savefig("Plots/{}/aa_dependence.svg".format(cath.rname), format='svg', bbox_inches='tight')
    plt.savefig("Plots/{}/aa_dependence.png".format(cath.rname),
                format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
