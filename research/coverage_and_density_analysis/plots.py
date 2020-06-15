import argparse
import numpy as np
from general import utils
import pathlib
import collections
from pprint import pprint
from nltk import word_tokenize
from nltk import sent_tokenize
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def main(args):

    """
    --wcep1 ~/Desktop/WCEP/analysis/wcep_stats.C10.json \
    --cnn ~/Desktop/WCEP/analysis/cnn_stats.json \
    --multinews ~/Desktop/WCEP/analysis/multinews_stats.json
    """
    dir_ = pathlib.Path('/home/demian/Desktop/WCEP/analysis')

    cnn_stats = utils.readjson(dir_ / 'cnn_stats.json')
    mn_stats = utils.readjson(dir_ / 'multinews_stats.json')
    wcep_stats_orig = utils.readjson(dir_ / 'wcep_stats_original.json')
    wcep_stats_10 = utils.readjson(dir_ / 'wcep_stats.C10.json')
    #wcep_stats_50 = utils.readjson(dir_ / 'wcep_stats.C50.json')
    wcep_stats_100 = utils.readjson(dir_ / 'wcep_stats.C100.json')
    duc_stats = utils.readjson(dir_ / 'duc_stats.json')
    all_stats = [wcep_stats_orig, wcep_stats_10, wcep_stats_100, cnn_stats, mn_stats, duc_stats]
    colors = ['Reds', 'Reds', 'Reds', 'Greens', 'Blues', 'Purples']
    #colors = ['b', 'b', 'b', 'r', 'g', 'm', 'c']
    names = ['WCEP-original', 'WCEP-10', 'WCEP-100', 'CNN', 'MultiNews', 'DUC']
    #fig, ax = plt.subplots(1, 3, sharey=True)
    #plt.style.use('dark_background')

    fig, ax = plt.subplots(3, 2, sharey=True)

    plt.rcParams["patch.force_edgecolor"] = True

    n_to_coord = {
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (0, 1),
        4: (1, 1),
        5: (2, 1),
    }

    font = {'family': 'normal',
            'color': 'black',
            'weight': 'normal',
            'size': 11,
            }

    for n in range(6):
        name = names[n]
        i, j = n_to_coord[n]
        ax_i = ax[i, j]
        ax_i.set_facecolor('white')

        print('Dataset:', names[n])

        stats = all_stats[n]
        coverage = np.array(stats['coverage'])
        density = np.array(stats['density'])
        print('Cov:', min(coverage), max(coverage), np.mean(coverage), np.median(coverage))
        print('Dense:', min(density), max(density), np.mean(density), np.median(density))
        ax_i.text(0.1, 8, name, fontdict=font)
        ax_i.set_ylim((0.0, 10.0))
        ax_i.set_xlim((0.0, 1.0))
        #ax_i.scatter(coverage, density, c=colors[n])
        sns.kdeplot(
            coverage,
            density,
            ax=ax_i,
            cmap=colors[n],
            shade=True,
            shade_lowest=False,
        )

        # ax_i.patch.set_edgecolor('black')
        # ax_i.patch.set_linewidth('2')

        ax_i.patch.set_edgecolor('black')
        ax_i.patch.set_linewidth(0.8)

    ax[2, 0].set_xlabel('Extractive fragment coverage')
    ax[1, 0].set_ylabel('Extractive fragment density')
    # #
    # plt.rcParams["axes.edgecolor"] = "black"
    # plt.rcParams["axes.linewidth"] = 1
    # sns.set_style("white")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cnn')
    # parser.add_argument('--wcep')
    # parser.add_argument('--multinews')
    main(parser.parse_args())
