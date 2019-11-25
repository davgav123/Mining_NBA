#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def prepare_data():
    df = pd.read_csv(Path('../data/shooting_by_zone_2018_19_data.csv'))

    # there are players who attempted ZERO shots from some zones and they will have NaN values thare
    # replace them with zeros because KMeams won't work otherwise
    df.fillna(0, inplace=True)

    # eliminate right/left corner shots data, but leave corner shots data in total
    df = df.drop(columns=['FGM_left_corner_three', 'FGA_left_corner_three', 'FG%_left_corner_three'])
    df = df.drop(columns=['FGM_right_corner_three', 'FGA_right_corner_three', 'FG%_right_corner_three'])

    # delete players with less than 100 shots in season
    df = df[df.FGA_restricted + df.FGA_paint_non_ra + df.FGA_mid_range + df.FGA_corner_three + df.FGA_above_the_break_three >= 100]

    # delete percentage data
    df = df.drop(columns=['FG%_restricted', 'FG%_paint_non_ra', 'FG%_mid_range', 'FG%_corner_three', 'FG%_above_the_break_three'])

    # delete makes data
    df = df.drop(columns=['FGM_restricted', 'FGM_paint_non_ra', 'FGM_mid_range', 'FGM_corner_three', 'FGM_above_the_break_three'])

    # select players
    players = df['Player']

    # drop player name, team and age
    df = df.drop(columns=['Player', 'TEAM', 'AGE'])

    return df, players


def calculate_best_k(data, min_k=2, max_k=12):
    '''
        Calculating wich number of clusters is
        the optimal one using silhouette score
    '''

    scores = []
    # calculate silhouette for different number of clusters
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, max_iter=800).fit(data)
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
    data, names = prepare_data()
    names = names.values

    calculate_best_k(data)

    # k = 6
    k = 9
    kmeans = KMeans(n_clusters=k, max_iter=800).fit(data)

    print(kmeans.cluster_centers_)
    # print(kmeans.labels_)

    # check in which cluster clustered players belong
    for (i, j) in zip(names, kmeans.predict(data)):
        print(i, j)

    # count how many players are in each of the clusters
    print(Counter(kmeans.labels_))
