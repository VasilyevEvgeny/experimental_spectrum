import numpy as np
from numpy import log10, where
from matplotlib import pyplot as plt
from os import path
from datetime import datetime
from time import sleep


def plot_ifs_comparison(data, colors, labels, res_dir, log_scale=True):

    # get data
    lambdas_list = []
    ifs_list = []
    for obj in data:
        lambdas_list.extend(obj.get_lambdas_list())
        ifs_list.extend(obj.get_ifs_list())

    # check shapes
    assert len(lambdas_list) == len(ifs_list)
    n = len(lambdas_list)

    min_val, max_val = +np.inf, -np.inf
    for ifs in ifs_list:
        min_val = min(min_val, np.min(ifs))
        max_val = max(max_val, np.max(ifs))

    if log_scale:
        max_log_val, min_log_val = 0, -2
    else:
        max_lin_val, min_lin_val = 1, 0

    #
    # figure
    #
    fig = plt.figure(figsize=(20, 10))

    # plot in cycle
    for i in range(n):
        lambdas = lambdas_list[i]
        ifs = ifs_list[i]

        # calculate spectrum integral
        spectrum_integral = 0.0
        for j in range(len(ifs)):
            spectrum_integral += ifs[j]
        print('{}: {}'.format(labels[i], spectrum_integral))

        if log_scale:
            # normalize and logarithm
            lowest_level = max_val * 10**-2
            ifs[where(ifs < lowest_level)] = lowest_level
            ifs = log10(ifs / max_val)
        else:
            # normalize
            ifs = ifs / max_val

        lambdas = [e / 10 ** 3 for e in lambdas]

        plt.plot(lambdas, ifs, linewidth=7, linestyle='solid', color=colors[i], label=labels[i], alpha=0.7)

    if log_scale:
        delta = 0.1 * (max_log_val - min_log_val)
        plt.ylim([min_log_val - delta, max_log_val + delta])
    else:
        delta = 0.1 * (max_lin_val - min_lin_val)
        plt.ylim([min_lin_val - delta, max_lin_val + delta])

    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.xlabel('$\mathbf{\lambda}$, $\mathbf{\mu}$m', fontsize=55, fontweight='bold')
    ylabel = 'lg(S/S$\mathbf{_{max}}$)' if log_scale else 'S/S$\mathbf{_{max}}$'
    plt.ylabel(ylabel, fontsize=55, fontweight='bold')

    plt.grid(linewidth=4, linestyle='dotted', color='gray', alpha=0.5)

    plt.legend(fontsize=30)

    bbox = fig.bbox_inches.from_bounds(0, -0.4, 19, 10)

    cur_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(path.join(res_dir, '{}_ifs_comparison.png'.format(cur_datetime)), bbox_inches=bbox)
    plt.close()

    sleep(1)
