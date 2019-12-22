#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def prepare_data():
    df = pd.read_csv(Path('../data/team_shooting_by_zone_per_game_data.csv'))
    
    # delete percentage data
    df = df.drop(columns=[
        'FG%_restricted', 'FG%_paint_non_ra', 'FG%_mid_range',
        'FG%_left_corner_three', 'FG%_right_corner_three', 'FG%_above_the_break_three'
    ])

    # delete makes data
    df = df.drop(columns=[
        'FGM_restricted', 'FGM_paint_non_ra', 'FGM_mid_range',
        'FGM_left_corner_three', 'FGM_right_corner_three', 'FGM_above_the_break_three'
    ])

    # add new column that will represent three pointers per game total
    df['FGA_three_pointer'] = df['FGA_left_corner_three'] + df['FGA_right_corner_three'] + df['FGA_above_the_break_three']

    # drop partial three pointers data
    df = df.drop(columns=['FGA_left_corner_three', 'FGA_right_corner_three', 'FGA_above_the_break_three'])

    # select teams
    teams = df['TEAM']

    # select season
    seasons = df['SEASON']

    # drop team name and season
    df = df.drop(columns=['TEAM', 'SEASON'])

    return df, teams, seasons


def calculate_best_k(data, min_k=2, max_k=12):
    '''
        Calculating which number of clusters is
        the optimal one using silhouette score
    '''

    scores = []
    # calculate silhouette for different number of clusters
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, max_iter=3500, n_init=35).fit(data)
        scores.append(silhouette_score(data, kmeans.labels_))

    # plot scores
    plt.plot(range(min_k, max_k + 1), scores, linewidth=5)
    plt.xticks(range(min_k, max_k + 1), fontsize=20)
    plt.yticks(fontsize=18)

    plt.xlabel('Number of clusters', fontsize=28)
    plt.ylabel('Silhouette score', fontsize=28)
    plt.grid()

    plt.show()

    # silhouette score standardized percent improvement
    # relative to the best possible score of 1
    standardized_scores = [1 - ((1 - x) / (1 - y)) for (x, y) in zip(scores[1:], scores[:-1])]

    # plot standardized scores
    plt.plot(range(min_k + 1, max_k + 1), standardized_scores, linewidth=5)
    plt.xticks(range(min_k + 1, max_k + 1), fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('Number of clusters', fontsize=28)
    plt.ylabel('percent of silhouette improvement', fontsize=28)
    plt.grid()

    plt.show()


if __name__ == "__main__":
    data, teams, seasons = prepare_data()
    teams = teams.values
    seasons = seasons.values

    teams = list(zip(teams, seasons))

    calculate_best_k(data)

    k = 5
    kmeans = KMeans(n_clusters=k, max_iter=3500, n_init=35).fit(data)

    # print centroids
    # print(kmeans.cluster_centers_)

    # count how many teams are in each of the clusters
    print(Counter(kmeans.labels_))

    # this is just for output
    labels = np.sort(np.unique(kmeans.labels_))
    team_prediction_pairs = list(zip(teams, kmeans.predict(data)))
    for label in labels:
        cc = kmeans.cluster_centers_[label]

        print('------------------------')
        print(f'cluster {label}:')
        print('------------------------')
        print(f'{cc} per game')
        print(f'{[np.around(81 * c) for c in cc]} totals')

        # calculate various percentages
        sum_of_shots = sum(cc)
        pct_of_ra_shots = (cc[0]) / sum_of_shots
        pct_of_midrange_shots = (cc[2]) / sum_of_shots
        pct_of_threes_shots = (cc[3]) / sum_of_shots
        pct_of_good_shots = (cc[3] + cc[0]) / sum_of_shots # 3P + RA

        print(f'\tPct of shots from Restricted area: {pct_of_ra_shots:.2f}')
        print(f'\tPct of shots from Mid-range: {pct_of_midrange_shots:.2f}')
        print(f'\tPct of shots from three-point: {pct_of_threes_shots:.2f}')
        print(f'\tPct of shots from RA + 3P: {pct_of_good_shots:.2f}')

        # print number of teams per season from cluster
        print(Counter([i[1] for (i, j) in team_prediction_pairs if label == j]))
        print()

        # print every (team, season) pair from cluster
        for (x, j) in team_prediction_pairs:
             if label == j:
                print(x)
        print()
