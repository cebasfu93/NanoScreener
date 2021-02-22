import numpy as np


# DEPRECATED
class Input:
    def __init__(self, sim, metric, mode):
        self.sim = sim
        self.metric = metric
        self.mode = mode
        self.i_space = np.linspace(1, 3, 75)  # eta
        self.plot_path = "S{}/{}/{}".format(sim[0], metric, mode)

        if "31" in sim or "32" in sim or "33" in sim:
            self.endtime = 10
            self.dist_suffix = "NVT"
            self.kwargs_rogs = {'xlim': (0, 10), 'xticks': (
                0, 5), 'ylim': (0.8, 1.7), 'yticks': (0.8, 1.1, 1.4)}
            self.kwargs_sasas = {'xlim': (0, 10), 'xticks': (
                0, 5), 'ylim': (90, 330), 'yticks': (90, 170, 250)}
        elif "34" in sim:
            self.endtime = 6
            self.dist_suffix = "NVT2"
            self.kwargs_rogs = {'xlim': (0, 6), 'xticks': (
                0, 2, 4), 'ylim': (0.9, 1.3), 'yticks': (0.9, 1.1)}
            self.kwargs_sasas = {'xlim': (0, 6), 'xticks': (
                0, 2, 4), 'ylim': (80, 160), 'yticks': (80, 120)}

        if mode == "Accuracy":
            self.opt_criter = 'accuracy'
            self.decide_criter = 'sensspec'
            self.plot_str = "acc"
        elif mode == "Sensspec":
            self.opt_criter = 'sensspec'
            self.decide_criter = 'accuracy'
            self.plot_str = "sensspec"

        if metric == 'Bye_time':
            if "31" in sim or "32" in sim or "33" in sim:
                self.j_space = np.linspace(0, 10, 100)
                xt = np.round(np.linspace(min(self.j_space), max(self.j_space), 6), 2)
                xlim = [0, 10]
            elif "34" in sim:
                self.j_space = np.linspace(0, 6, 100)
                xt = np.round(np.linspace(min(self.j_space), max(self.j_space), 7), 2)
                xlim = [0, 6]
            self.kwargs_paramsearch = {'xlabel': 'Depart time (ns)', 'xticks': np.linspace(0, len(self.j_space), 6)[:-1], 'xticklabels': xt[:-1],
                                       'ylabel': r"$\eta$ factor", 'yticks': np.linspace(0, len(self.i_space), 6)[:-1],
                                       'yticklabels': np.round(np.linspace(min(self.i_space), max(self.i_space), 6), 1),
                                       'path': self.plot_path, 'plot_str': self.plot_str}
            self.kwargs = {'xlabel': 'Depart time (ns)', 'xlim': xlim, 'xticks': xt,
                           'pdf_lim': (0, 2), 'pdf_ticks': np.arange(0, 2.1, 1.),
                           'distr_lim': (0, 1), 'distr_ticks': np.linspace(0, 1, 6),
                           'path': self.plot_path, 'plot_str': self.plot_str}

        elif metric == 'Bound_fraction':
            self.j_space = np.linspace(0, 1, 100)
            self.kwargs_paramsearch = {'xlabel': 'Bound fraction', 'xticks': np.linspace(0, len(self.j_space), 6)[:-1],
                                       'xticklabels': np.round(np.linspace(min(self.j_space), max(self.j_space), 6), 2)[:-1],
                                       'ylabel': r"$\eta$ factor", 'yticks': np.linspace(0, len(self.i_space), 6)[:-1],
                                       'yticklabels': np.round(np.linspace(min(self.i_space), max(self.i_space), 6), 1),
                                       'path': self.plot_path, 'plot_str': self.plot_str}
            self.kwargs = {'xlabel': 'Bound fraction', 'xlim': (0, 1), 'xticks': np.round(np.linspace(0, 1, 6), 1),
                           'pdf_lim': (0, 18), 'pdf_ticks': np.linspace(0, 18, 4, dtype='int'),
                           'distr_lim': (0, 8), 'distr_ticks': np.linspace(0, 8, 5),
                           'path': self.plot_path, 'plot_str': self.plot_str}

        elif metric == 'Weighted_fraction':
            self.j_space = np.linspace(0, 1, 100)
            self.kwargs_paramsearch = {'xlabel': 'Weighted fraction', 'xticks': np.linspace(0, len(self.j_space), 6)[:-1],
                                       'xticklabels': np.round(np.linspace(min(self.j_space), max(self.j_space), 6), 2)[:-1],
                                       'ylabel': r"$\eta$ factor", 'yticks': np.linspace(0, len(self.i_space), 6)[:-1],
                                       'yticklabels': np.round(np.linspace(min(self.i_space), max(self.i_space), 6), 1),
                                       'path': self.plot_path, 'plot_str': self.plot_str}
            self.kwargs = {'xlabel': 'Weigthed fraction', 'xlim': (0, 1), 'xticks': np.round(np.linspace(0, 1, 6), 1),
                           'pdf_lim': (0, 15), 'pdf_ticks': np.linspace(0, 15, 4, dtype='int'),
                           'distr_lim': (0, 8), 'distr_ticks': np.linspace(0, 8, 5),
                           'path': self.plot_path, 'plot_str': self.plot_str}

    def update_metric_params(self, times, data, th):
        if self.metric == 'Weighted_fraction':
            self.metric_params = (th, times, self.endtime, data)
        else:
            self.metric_params = (times, data)
