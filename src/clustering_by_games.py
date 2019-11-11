#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


def clustering_by_games_played(columns, num_clusters=4, scale=False, show=False):
    df = pd.read_csv(Path('../data/per_game_data.csv'))

    # just 2018-19 seaosn
    df = df[df.season.isin([2019])]
    # take just selected columns
    df = df[columns]

    print('************************************************')
    print(f'Selected columns are {columns}. Number of clusters is {num_clusters}.')

    # scale to [0, 1] if needed
    if scale:
        df = MinMaxScaler().fit_transform(df)

    # kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=200).fit(df)
    print(f'KMeans silhouette score: {silhouette_score(df, kmeans.labels_)}')
    if show:
        show_clusters(df, columns, kmeans, 'KMeans', scale)

    # hierarchichal clustering, single linkage
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='single').fit(df)
    print(f'Agglomerative (single) silhouette score: {silhouette_score(df, agglomerative.labels_)}')
    if show:
        show_clusters(df, columns, agglomerative, 'Agglomerative, single linkage', scale)

    # hierarchichal clustering, complete linkage
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete').fit(df)
    print(f'Agglomerative (complete) silhouette score: {silhouette_score(df, agglomerative.labels_)}')
    if show:
        show_clusters(df, columns, agglomerative, 'Agglomerative, complete linkage', scale)

    # hierarchichal clustering, average linkage
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='average').fit(df)
    print(f'Agglomerative (average) silhouette score: {silhouette_score(df, agglomerative.labels_)}')
    if show:
        show_clusters(df, columns, agglomerative, 'Agglomerative, average linkage', scale)

    # hierarchichal clustering, Ward's linkage
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(df)
    print(f'Agglomerative (ward) silhouette score: {silhouette_score(df, agglomerative.labels_)}')
    if show:
        show_clusters(df, columns, agglomerative, 'Agglomerative, Ward\'s linkage', scale)


def show_clusters(data, columns, clst, title, scale=False):
    title_size = 32
    label_size = 24
    ticks_size = 20

    # plot
    plt.figure(figsize=(8, 8))

    if scale:
        plt.scatter(data[:, 0], data[:, 1], c=clst.labels_)
    else:
        plt.scatter(data[columns[0]], data[columns[1]], c=clst.labels_)

    plt.title(title, fontsize=title_size)
    plt.xlabel(columns[0], fontsize=label_size)
    plt.ylabel(columns[1], fontsize=label_size)

    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)

    plt.show()


if __name__ == "__main__":
    # clustering_by_games_played(columns=['G', 'GS'], num_clusters=3)
    # clustering_by_games_played(columns=['G', 'MP'], num_clusters=3)
    clustering_by_games_played(columns=['G', 'GS', 'MP'], num_clusters=3)

    print('Scale to [0, 1]')
    # clustering_by_games_played(columns=['G', 'GS'], num_clusters=3, scale=True)
    # clustering_by_games_played(columns=['G', 'MP'], num_clusters=3, scale=True)
    clustering_by_games_played(columns=['G', 'GS', 'MP'], num_clusters=3, scale=True)

    # clustering_by_games_played(columns=['G', 'GS'], num_clusters=4)
    # clustering_by_games_played(columns=['G', 'MP'], num_clusters=4)
    clustering_by_games_played(columns=['G', 'GS', 'MP'], num_clusters=4)

    print('Scale to [0, 1]')
    # clustering_by_games_played(columns=['G', 'GS'], num_clusters=4, scale=True)
    # clustering_by_games_played(columns=['G', 'MP'], num_clusters=4, scale=True)
    clustering_by_games_played(columns=['G', 'GS', 'MP'], num_clusters=4, scale=True, show=True)
