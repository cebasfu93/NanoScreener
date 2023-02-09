import numpy as np
from DEPENDENCIES.constants import *
import DEPENDENCIES.processing as proc
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_distributions(j_space, best_j, bind_scores, unbind_scores, kwargs={}):
    """
    j_space is a 1D array with all the possible free energy thresholds segregating active from inactive dyads
    best_j is the best (chosen) threshold
    bind_scores is a 1D array with the computed binding energies of the dyads classified as 'active'
    unbind_scores is a 1D array with the computed binding energies of the dyads classified as 'inactive'
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes one figure with two plots:
    1) Probability distribution of binding free energy separated into 'classified as active' and 'classified as inactive' dyads
    2) Cumulative probability of finding a particular binding free energy in an (in)active dyad
    """
    fig, axs = plt.subplots(
        figsize=kwargs.get("figsize", (11, 8)),
        ncols=1,
        nrows=2,
        gridspec_kw={"hspace": 0.35},
    )

    for ax in axs:
        ax.tick_params(labelsize=Z, size=6, width=1.4)
        ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        xt = ax.get_xticks()
        ax.set_xticklabels(kwargs.get("xticklabels", xt))
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
        ax.grid()
        ax.axvline(best_j, ymin=0, ymax=1, c="k", lw=2, ls="-", zorder=100)

    mu, sigma = norm.fit(unbind_scores)
    pdf = proc.gaussian(j_space, mu, sigma)
    axs[0].plot(j_space, pdf, color="b", lw=3.5, label="Inactive")
    n, bins, _ = axs[0].hist(unbind_scores, bins=30, alpha=0.3, density=True, color="b")
    axs[1].plot(
        np.append(bins[:-1], [axs[1].get_xlim()[1]]),
        np.append(np.cumsum(n * (bins[1] - bins[0])), [1]),
        color="b",
        lw=3.5,
    )

    mu, sigma = norm.fit(bind_scores)
    pdf = proc.gaussian(j_space, mu, sigma)
    axs[0].plot(j_space, pdf, color="orange", lw=3.5, label="Active")
    n, bins, _ = axs[0].hist(
        bind_scores, bins=30, alpha=0.3, density=True, color="orange"
    )
    axs[1].plot(
        np.append(bins[:-1], [axs[1].get_xlim()[1]]),
        np.append(np.cumsum(n * (bins[1] - bins[0])), [1]),
        color="orange",
        lw=3.5,
    )

    axs[0].set_ylabel("Probability\ndensity", fontsize=Z)
    axs[0].set_ylim(kwargs.get("distr_lim", (0, 1)))
    axs[0].set_yticks(kwargs.get("distr_ticks", (0, 1)))
    axs[0].legend(loc="upper right", fontsize=Z - 2, ncol=1)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("Cumulative\nprobability", fontsize=Z)
    axs[1].set_yticks([0, 0.50, 1.00])
    axs[1].set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)
    if kwargs.get("svg", False):
        plt.savefig(
            kwargs.get("path", ".")
            + "/Distributions_"
            + kwargs.get("plot_str", "def")
            + ".svg",
            format="svg",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            kwargs.get("path", ".")
            + "/Distributions_"
            + kwargs.get("plot_str", "def")
            + ".png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.show()
    plt.close()


def plot_score_distribution(best_j, j_space, data=DATA, kwargs={}):
    """
    j_space is a 1D array with all the possible free energy thresholds segregating active from inactive dyads
    best_j is the best (chosen) threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes one figure with three plots:
    1) Experimental (log (K)) vs computed (Delta G) affinity for each dyad
    2) Probability distribution of Delta G for each dyad
    3) 1D plot marking the mean Delta G of each dyad
    """
    smooth_j = np.linspace(min(j_space), max(j_space), 400)
    fig, axs = plt.subplots(
        figsize=kwargs.get("figsize", (11, 8)),
        sharex=True,
        ncols=1,
        nrows=3,
        gridspec_kw={"hspace": 0.2, "height_ratios": [3, 1, 0.25]},
    )
    for ax in axs:
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=2, ls="-", zorder=100)
        ax.grid(alpha=0.5)
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
        ax.tick_params(labelsize=Z, size=6, width=1.4)

    axs[0].axhline(y=LOGACT_THRES, c="k", lw=2, ls="-", zorder=100)
    axs[0].set_yticks([0, 2, 3, 4, 5, 6, 7])
    axs[0].set_ylim(-0.5, 7)
    axs[0].set_ylabel("Log(Activity)", fontsize=Z)
    axs[1].set_ylim(kwargs.get("pdf_lim", (0, 1)))
    axs[1].set_ylabel("Probability\ndensity", fontsize=Z)
    axs[1].set_yticks(kwargs.get("pdf_ticks", np.round(np.linspace(0, 1, 6), 2)))
    axs[2].set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)
    axs[2].set_yticks([])
    axs[2].set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
    xt = axs[2].get_xticks()
    axs[2].set_xticklabels(kwargs.get("xticklabels", xt))

    for act_log, c, score in zip(data.Act_log, data.Color, data.Score):
        axs[0].errorbar(score, [act_log] * len(score), c=c, fmt="o", ms=5, alpha=0.2)
        axs[0].errorbar(
            np.mean(score),
            act_log,
            xerr=np.std(score) / (len(score) ** 0.5),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.6,
            ecolor="k",
            capsize=2,
        )  # boltz_mean
        axs[1].plot(
            smooth_j,
            proc.gaussian(
                smooth_j, np.mean(score), np.std(score) / (len(score)) ** 0.5
            ),
            color=c,
            alpha=0.7,
            lw=2.5,
        )  # ste
        axs[2].axvline(x=np.mean(score), ymin=0, ymax=1, color=c, lw=2.5)

    if kwargs.get("svg", False):
        print(
            kwargs.get("path", ".")
            + "/scoredistr_"
            + kwargs.get("plot_str", "def")
            + ".svg"
        )
        plt.savefig(
            kwargs.get("path", ".")
            + "/scoredistr_"
            + kwargs.get("plot_str", "def")
            + ".svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
    else:
        plt.savefig(
            kwargs.get("path", ".")
            + "/scoredistr_"
            + kwargs.get("plot_str", "def")
            + ".png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.show()
    plt.close()


def plot_score_rank(best_j, j_space, data=DATA, kwargs={}):
    """
    j_space is a 1D array with all the possible free energy thresholds segregating active from inactive dyads
    best_j is the best (chosen) threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes one figure with three plots:
    1) Experimental vs computed ranking for each dyad
    2) Probability distribution of Delta G for each dyad
    3) 1D plot marking the mean Delta G of each dyad
    """
    fig = plt.figure(figsize=(8, 10))
    ax = plt.axes()
    ax2 = ax.twinx()
    ax.set_ylim(len(data), -1)
    ax2.set_ylim(len(data), -1)
    ax2.set_yticks(data.Act_Rank)
    ax2.set_yticklabels(data.index, fontsize=Z - 4)
    ax2.set_xticks([])
    ax2.grid(alpha=0.3)
    ax.set_xlim(kwargs.get("xlim", (0, 1)))
    ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
    ax.tick_params(labelsize=Z)
    ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)
    ax.set_ylabel("Experimental ranking", fontsize=Z)
    ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=2, ls="-", zorder=100)
    ax.axhline(
        y=data.Act_Rank[np.where(~data.Active)[0][-1]] - 0.5,
        c="k",
        lw=2,
        ls="-",
        zorder=100,
    )
    for i, (score, rank, c) in enumerate(zip(data.Score, data.Act_Rank, data.Color)):
        ax.errorbar([score], [rank], c=c, fmt="o", ms=5, alpha=0.5)
        ax.errorbar(
            np.mean(score),
            [rank],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=1.3,
            ecolor="k",
            capsize=2,
        )
    plt.savefig(
        kwargs.get("path", ".")
        + "/scoredistr_rank_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_np_dependence(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each unique nanoparticle. It shows the
    experimental (log (K)) vs computed (Delta G) affinity for each dyad,
    highlighting the dyads with a certain nanoparticle
    """
    fig, axs = plt.subplots(
        figsize=(20, 26), ncols=3, nrows=4, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for a, (ax, NN) in enumerate(zip(axs.flatten(), np.unique(data.Nanoparticle))):
        ax.set_ylim(-0.5, 7)
        ax.set_title("NP-{}".format(NN), fontsize=Z + 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(y=LOGACT_THRES, xmin=0, xmax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.tick_params(labelsize=Z)
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
        for i, (score, act_log, c, nn) in enumerate(
            zip(data.Score, data.Act_log, data.Color, data.Nanoparticle)
        ):
            if nn == NN:
                al1 = 0.6
                al2 = 1.0
            else:
                al1 = 0.03
                al2 = 0.15
            ax.errorbar(score, len(score) * [act_log], c=c, fmt="o", ms=5, alpha=al1)
            ax.errorbar(
                np.mean(score),
                [act_log],
                xerr=np.std(score),
                c=c,
                fmt="o",
                ms=7,
                mec="k",
                mew=1.5,
                zorder=50,
                elinewidth=0.8,
                ecolor="k",
                capsize=2,
                alpha=al2,
            )
    for ax in axs[:, 0]:
        ax.set_yticks(np.linspace(0, 6, 7))
        ax.set_ylabel("Log(Activity)", fontsize=Z)
    for a, ax in enumerate(axs[-1, :]):
        if (a - 2) % 4 == 0:
            ax.set_xticks([])
        else:
            ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)

    axs[-1, -1].set_yticks([])
    plt.savefig(
        kwargs.get("path", ".")
        + "/np_dependence_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_np_dependence_rank(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each unique nanoparticle. It shows the
    experimental vs computed rankings for each dyad,
    highlighting the dyads with a certain nanoparticle
    """
    fig, axs = plt.subplots(
        figsize=(20, 32), ncols=3, nrows=4, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for a, (ax, NN) in enumerate(zip(axs.flatten(), np.unique(data.Nanoparticle))):
        ax2 = ax.twinx()
        ax2.set_ylim(31, -1)
        ax2.set_yticks(data.Act_Rank)
        ax2.grid(alpha=0.2)
        if a % 3 == 2:
            ax2.set_yticklabels(data.index)
        else:
            ax2.set_yticklabels([])
        ax.set_ylim(31, -1)
        ax.set_title("NP-{}".format(NN), fontsize=Z + 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axhline(
            y=data.Act_Rank[np.where(~data.Active)[0][-1]] - 0.5,
            c="k",
            lw=0.9,
            ls="-",
            zorder=100,
        )
        ax.tick_params(labelsize=Z)
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
        for i, (score, rank, c, nn) in enumerate(
            zip(data.Score, data.Act_Rank, data.Color, data.Nanoparticle)
        ):
            if nn == NN:
                al1 = 0.6
                al2 = 1.0
            else:
                al1 = 0.03
                al2 = 0.15
            ax.errorbar(score, len(score) * [rank], c=c, fmt="o", ms=5, alpha=al1)
            ax.errorbar(
                np.mean(score),
                [rank],
                xerr=np.std(score),
                c=c,
                fmt="o",
                ms=7,
                mec="k",
                mew=1.5,
                zorder=50,
                elinewidth=0.8,
                ecolor="k",
                capsize=2,
                alpha=al2,
            )
    for ax in axs[:, 0]:
        ax.set_yticks(np.linspace(0, 30, 7))
        ax.set_ylabel("Experimental ranking", fontsize=Z)
    for a, ax in enumerate(axs[-1, :]):
        if (a - 2) % 4 == 0:
            ax.set_xticks([])
        else:
            ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)

    axs[-1, -1].set_yticks([])
    ax2 = axs[-1, -1].twinx()
    ax2.set_ylim(-1, 31)
    ax2.set_yticks(data.Act_Rank)
    ax2.grid(alpha=0.2)
    ax2.set_yticklabels(data.index)
    plt.savefig(
        kwargs.get("path", ".")
        + "/np_dependence_rank_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_an_dependence(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each unique analyte. It shows the
    experimental (log (K)) vs computed (Delta G) affinity for each dyad,
    highlighting the dyads with a certain analyte
    """
    fig, axs = plt.subplots(
        figsize=(16, 39), ncols=3, nrows=6, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for a, (ax, AA) in enumerate(zip(axs.flatten(), np.unique(data.Analyte))):
        ax.set_ylim(-0.5, 7)
        ax.set_title("A-{}".format(AA), fontsize=Z + 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(y=LOGACT_THRES, xmin=0, xmax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.tick_params(labelsize=Z)
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
        for i, (score, act_log, c, aa) in enumerate(
            zip(data.Score, data.Act_log, data.Color, data.Analyte)
        ):
            if aa == AA:
                al1 = 0.6
                al2 = 1.0
            else:
                al1 = 0.03
                al2 = 0.15
            ax.errorbar(score, len(score) * [act_log], c=c, fmt="o", ms=5, alpha=al1)
            ax.errorbar(
                np.mean(score),
                [act_log],
                xerr=np.std(score),
                c=c,
                fmt="o",
                ms=7,
                mec="k",
                mew=1.5,
                zorder=50,
                elinewidth=0.8,
                ecolor="k",
                capsize=2,
                alpha=al2,
            )
    for ax in axs[:, 0]:
        ax.set_yticks(np.linspace(0, 6, 7))
        ax.set_ylabel("Log(Activity)", fontsize=Z)
    for a, ax in enumerate(axs[-1, :]):
        if (a - 2) % 4 == 0:
            ax.set_xticks([])
        else:
            ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)

    axs[-1, -2].set_yticks([])
    axs[-1, -1].set_yticks([])
    plt.savefig(
        kwargs.get("path", ".")
        + "/an_dependence_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_an_dependence_rank(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each unique analyte. It shows the
    experimental vs computed rankings for each dyad,
    highlighting the dyads with a certain analyte
    """
    fig, axs = plt.subplots(
        figsize=(20, 48), ncols=3, nrows=6, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for a, (ax, AA) in enumerate(zip(axs.flatten(), np.unique(data.Analyte))):
        ax2 = ax.twinx()
        ax2.set_ylim(31, -1)
        ax2.set_yticks(data.Act_Rank)
        ax2.grid(alpha=0.2)
        if a % 3 == 2:
            ax2.set_yticklabels(data.index)
        else:
            ax2.set_yticklabels([])
        ax.set_ylim(31, -1)
        ax.set_title("A-{}".format(AA), fontsize=Z + 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axhline(
            y=data.Act_Rank[np.where(~data.Active)[0][-1]] - 0.5,
            c="k",
            lw=0.9,
            ls="-",
            zorder=100,
        )
        ax.tick_params(labelsize=Z)
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
        for i, (score, rank, c, aa) in enumerate(
            zip(data.Score, data.Act_Rank, data.Color, data.Analyte)
        ):
            if aa == AA:
                al1 = 0.6
                al2 = 1.0
            else:
                al1 = 0.03
                al2 = 0.15
            ax.errorbar(score, len(score) * [rank], c=c, fmt="o", ms=5, alpha=al1)
            ax.errorbar(
                np.mean(score),
                [rank],
                xerr=np.std(score),
                c=c,
                fmt="o",
                ms=7,
                mec="k",
                mew=1.5,
                zorder=50,
                elinewidth=0.8,
                ecolor="k",
                capsize=2,
                alpha=al2,
            )
    for ax in axs[:, 0]:
        ax.set_yticks(np.linspace(0, 30, 7))
        ax.set_ylabel("Experimental ranking", fontsize=Z)
    for a, ax in enumerate(axs[-1, :]):
        if (a - 2) % 4 == 0:
            ax.set_xticks([])
        else:
            ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)

    axs[-1, -2].set_yticks([])
    axs[-1, -1].set_yticks([])
    ax2 = axs[-1, -1].twinx()
    ax2.set_ylim(-1, 31)
    ax2.set_yticks(data.Act_Rank)
    ax2.grid(alpha=0.2)
    ax2.set_yticklabels(data.index)
    plt.savefig(
        kwargs.get("path", ".")
        + "/an_dependence_rank_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_npcharge_dependence(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each kind of charge (anionic, neutral, cationic) in a nanoparticle.
    It shows the experimental (log (K)) vs computed (Delta G) affinity for each dyad,
    highlighting the dyads with a certain sign of charge in the nanoparticle
    """
    titles = ["Charged NPs", "Neutral NPs"]
    fig, axs = plt.subplots(
        figsize=(10, 6), ncols=2, nrows=1, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for ax, title in zip(axs, titles):
        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)
        ax.set_ylim(-0.5, 7)
        ax.set_title(title, fontsize=Z + 4)
        ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_yticks([])
        ax.tick_params(labelsize=Z)
        ax.axhline(y=LOGACT_THRES, xmin=0, xmax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
    for i, (score, act_log, c, nn) in enumerate(
        zip(data.Score, data.Act_log, data.Color, data.Nanoparticle)
    ):
        if nn in charged_nps:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[0].errorbar(score, len(score) * [act_log], c=c, fmt="o", ms=5, alpha=al1)
        axs[0].errorbar(
            np.mean(score),
            [act_log],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    for i, (score, act_log, c, nn) in enumerate(
        zip(data.Score, data.Act_log, data.Color, data.Nanoparticle)
    ):
        if nn not in charged_nps:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[1].errorbar(score, len(score) * [act_log], c=c, fmt="o", ms=5, alpha=al1)
        axs[1].errorbar(
            np.mean(score),
            [act_log],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )

    axs[0].set_yticks(np.linspace(0, 6, 7))
    axs[0].set_ylabel("Log(Activity)", fontsize=Z)
    plt.savefig(
        kwargs.get("path", ".")
        + "/npcharge_dependence_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_npcharge_dependence_rank(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each kind of charge (anionic, neutral, cationic) in a nanoparticle.
    It shows the experimental vs computed rankings for each dyad,
    highlighting the dyads with a certain sign of charge in the nanoparticle
    """
    titles = ["Charged NPs", "Neutral NPs"]
    fig, axs = plt.subplots(
        figsize=(12, 8), ncols=2, nrows=1, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for ax, title in zip(axs, titles):
        ax2 = ax.twinx()
        ax2.set_ylim(31, -1)
        ax2.set_yticks(DATA.Act_Rank)
        ax2.set_yticklabels([])
        ax2.grid(alpha=0.2)

        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)
        ax.set_ylim(31, -1)
        ax.set_title(title, fontsize=Z + 4)
        ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_yticks([])
        ax.tick_params(labelsize=Z)
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axhline(
            y=data.Act_Rank[np.where(~data.Active)[0][-1]] - 0.5,
            c="k",
            lw=0.9,
            ls="-",
            zorder=100,
        )
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
    ax2.set_yticklabels(DATA.index)
    for i, (score, rank, c, nn) in enumerate(
        zip(data.Score, data.Act_Rank, data.Color, data.Nanoparticle)
    ):
        if nn in charged_nps:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[0].errorbar(score, len(score) * [rank], c=c, fmt="o", ms=5, alpha=al1)
        axs[0].errorbar(
            np.mean(score),
            [rank],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    for i, (score, rank, c, nn) in enumerate(
        zip(data.Score, data.Act_Rank, data.Color, data.Nanoparticle)
    ):
        if nn not in charged_nps:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[1].errorbar(score, len(score) * [rank], c=c, fmt="o", ms=5, alpha=al1)
        axs[1].errorbar(
            np.mean(score),
            [rank],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    axs[0].set_yticks(np.linspace(0, 30, 7))
    axs[0].set_ylabel("Experimental ranking", fontsize=Z)
    plt.savefig(
        kwargs.get("path", ".")
        + "/npcharge_dependence_rank_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_ancharge_dependence(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each kind of charge (anionic, neutral, cationic) in a analyte.
    It shows the experimental (log (K)) vs computed (Delta G) affinity for each dyad,
    highlighting the dyads with a certain sign of charge in the analyte
    """
    titles = ["Cationic As", "Anionic As", "Zwitterionic As"]
    fig, axs = plt.subplots(
        figsize=(15, 6), ncols=3, nrows=1, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for ax, title in zip(axs, titles):
        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)
        ax.set_ylim(-0.5, 7)
        ax.set_title(title, fontsize=Z + 4)
        ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_yticks([])
        ax.tick_params(labelsize=Z)
        ax.axhline(y=LOGACT_THRES, xmin=0, xmax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
    for i, (score, act_log, c, aa) in enumerate(
        zip(data.Score, data.Act_log, data.Color, data.Analyte)
    ):
        if aa in cationic_a:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[0].errorbar(score, len(score) * [act_log], c=c, fmt="o", ms=5, alpha=al1)
        axs[0].errorbar(
            np.mean(score),
            [act_log],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    for i, (score, act_log, c, aa) in enumerate(
        zip(data.Score, data.Act_log, data.Color, data.Analyte)
    ):
        if aa in anionic_a:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[1].errorbar(score, len(score) * [act_log], c=c, fmt="o", ms=5, alpha=al1)
        axs[1].errorbar(
            np.mean(score),
            [act_log],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    for i, (score, act_log, c, aa) in enumerate(
        zip(data.Score, data.Act_log, data.Color, data.Analyte)
    ):
        if aa not in cationic_a and aa not in anionic_a:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[2].errorbar(score, len(score) * [act_log], c=c, fmt="o", ms=5, alpha=al1)
        axs[2].errorbar(
            np.mean(score),
            [act_log],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )

    axs[0].set_yticks(np.linspace(0, 6, 7))
    axs[0].set_ylabel("Log(Activity)", fontsize=Z)
    plt.savefig(
        kwargs.get("path", ".")
        + "/ancharge_dependence_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_ancharge_dependence_rank(best_j, data=DATA, kwargs={}):
    """
    best_j is the best (chosen) free energy threshold
    data is a dataframe with all the information of the receptor-analyte dyad (see PMF_analysis.ipynb)
    The kwargs dictionary is for plotting, for the most part. Most keywords are those given to matplotlib

    It makes a grided figure with one plot for each kind of charge (anionic, neutral, cationic) in a analyte.
    It shows the experimental vs computed ranking for each dyad,
    highlighting the dyads with a certain sign of charge in the analyte
    """
    titles = ["Cationic As", "Anionic As", "Zwitterionic As"]
    fig, axs = plt.subplots(
        figsize=(18, 8), ncols=3, nrows=1, gridspec_kw={"hspace": 0.1, "wspace": 0.03}
    )
    for ax, title in zip(axs, titles):
        ax2 = ax.twinx()
        ax2.set_ylim(31, -1)
        ax2.set_yticks(data.Act_Rank)
        ax2.set_yticklabels([])
        ax2.grid(alpha=0.2)

        ax.set_xlabel(kwargs.get("xlabel", "j space"), fontsize=Z)
        ax.set_ylim(31, -1)
        ax.set_title(title, fontsize=Z + 4)
        ax.set_xticks(kwargs.get("xticks", np.round(np.linspace(0, 1, 6), 1)))
        ax.set_yticks([])
        ax.tick_params(labelsize=Z)
        ax.axvline(x=best_j, ymin=0, ymax=1, c="k", lw=0.9, ls="-", zorder=100)
        ax.axhline(
            y=data.Act_Rank[np.where(~data.Active)[0][-1]] - 0.5,
            c="k",
            lw=0.9,
            ls="-",
            zorder=100,
        )
        ax.set_xlim(kwargs.get("xlim", (0, 1)))
    ax2.set_yticklabels(data.index)
    for i, (score, rank, c, aa) in enumerate(
        zip(data.Score, data.Act_Rank, data.Color, data.Analyte)
    ):
        if aa in cationic_a:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[0].errorbar(score, len(score) * [rank], c=c, fmt="o", ms=5, alpha=al1)
        axs[0].errorbar(
            np.mean(score),
            [rank],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    for i, (score, rank, c, aa) in enumerate(
        zip(data.Score, data.Act_Rank, data.Color, data.Analyte)
    ):
        if aa in anionic_a:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[1].errorbar(score, len(score) * [rank], c=c, fmt="o", ms=5, alpha=al1)
        axs[1].errorbar(
            np.mean(score),
            [rank],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    for i, (score, rank, c, aa) in enumerate(
        zip(data.Score, data.Act_Rank, data.Color, data.Analyte)
    ):
        if aa not in cationic_a and aa not in anionic_a:
            al1 = 0.6
            al2 = 1.0
        else:
            al1 = 0.03
            al2 = 0.15
        axs[2].errorbar(score, len(score) * [rank], c=c, fmt="o", ms=5, alpha=al1)
        axs[2].errorbar(
            np.mean(score),
            [rank],
            xerr=np.std(score),
            c=c,
            fmt="o",
            ms=7,
            mec="k",
            mew=1.5,
            zorder=50,
            elinewidth=0.8,
            ecolor="k",
            capsize=2,
            alpha=al2,
        )
    axs[0].set_yticks(np.linspace(0, 30, 7))
    axs[0].set_ylabel("Experimental ranking", fontsize=Z)
    plt.savefig(
        kwargs.get("path", ".")
        + "/ancharge_dependence_rank_"
        + kwargs.get("plot_str", "def")
        + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("This statement will be executed only if this script is called directly")
