#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def jokic_stats_plot(path, stats, title):
    df = pd.read_csv(path)
    # jokic
    df = df[df.Player == 'Nikola Jokić']

    # filter stats
    df = df[stats]
    # print(df)

    # filter out 'Age' from the stats
    stats = stats[1:]

    # plot ordinary plots
    for stat in stats:
        plt.plot(df['Age'], df[stat], linewidth=5)

    # plot scatter plots
    for stat in stats:
        plt.scatter(df['Age'], df[stat], s=80)

    plt.xticks(df['Age'], fontsize=24)
    plt.xlabel('Age', fontsize=30)

    plt.yticks(fontsize=24)
    plt.ylabel('Stat value', fontsize=30)

    plt.grid(axis='y')
    plt.title(title, fontsize=36)    
    plt.legend(stats, loc=0, fontsize=22)
    plt.show()


def draw_one_boxplot(df, df_jokic, stat, lbls, title):
    plt.boxplot(df[stat], notch=True, showfliers=False)
    plt.scatter([1 for i in range(len(df[stat]))], df[stat], c='red')
    jokic = plt.scatter([1], df_jokic[stat], c='blue', s=80)

    plt.xticks([1], labels=lbls, fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid(axis='y')
    plt.legend(handles=[jokic], labels=['Jokić'], fontsize=22)
    plt.title(title, fontsize=36)
    plt.show()


def draw_two_boxplots(df, df_jokic, stats, lbls, title):
    plt.boxplot([df[stats[0]], df[stats[1]]], notch=True, showfliers=False)
    plt.scatter([1 for i in range(len(df[stats[0]]))], df[stats[0]], c='red')
    plt.scatter([2 for i in range(len(df[stats[1]]))], df[stats[1]], c='red')
    jokic = plt.scatter([1, 2], [df_jokic[stats[0]].values, df_jokic[stats[1]].values], c='blue', s=80)

    plt.xticks([1, 2], labels=lbls, fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid(axis='y')
    plt.legend(handles=[jokic], labels=['Jokić'], fontsize=22)
    plt.title(title, fontsize=36)
    plt.show()


def draw_three_boxplots(df, df_jokic, stats, lbls, title):
    plt.boxplot([df[stats[0]], df[stats[1]], df[stats[2]]], notch=True, showfliers=False)
    plt.scatter([1 for i in range(len(df[stats[0]]))], df[stats[0]], c='red')
    plt.scatter([2 for i in range(len(df[stats[1]]))], df[stats[1]], c='red')
    plt.scatter([3 for i in range(len(df[stats[2]]))], df[stats[2]], c='red')
    jokic = plt.scatter([1, 2, 3], [df_jokic[stats[0]].values, df_jokic[stats[1]].values, df_jokic[stats[2]].values], c='blue', s=80)

    plt.xticks([1, 2, 3], labels=lbls, fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid(axis='y')
    plt.legend(handles=[jokic], labels=['Jokić'], fontsize=22)
    plt.title(title, fontsize=36)
    plt.show()


def compare_jokic_to_centers_traditional():
    # first, per game stats are compared
    df = pd.read_csv(Path('../data/per_game_data.csv'))

    # centers
    df = df[df.Pos.isin(['C', 'C-PF'])]

    # players from 2018-19 season
    df = df[df.season == 2019]

    # Centers with important roles on their teams
    df = df[df.G > 35]
    df = df[df.MP > 15.0]

    # jokic
    dfj = df[df.Player == 'Nikola Jokić']

    # box plot PTS
    draw_one_boxplot(
        df,
        dfj,
        'PTS',
        ['PTS/G'],
        'Points per game'
    )

    # box plot DRB, ORB and TRB
    draw_three_boxplots(
        df,
        dfj,
        ['DRB', 'ORB', 'TRB'],
        ['DRB/G', 'ORB/G', 'TRB/G'],
        'Rebounds per game'
    )

    # box plot AST and TOV
    draw_two_boxplots(
        df,
        dfj,
        ['AST', 'TOV'],
        ['AST/G', 'TOV/G'],
        'AST and TOV per game'
    )


def compare_jokic_to_centers_advanced():
    df = pd.read_csv(Path('../data/advanced_data.csv'))

    # centers
    df = df[df.Pos.isin(['C', 'C-PF'])]

    # players from 2018-19 season
    df = df[df.season == 2019]

    # Centers with important roles on their teams
    df = df[df.G > 35]
    df = df[df.MP > (15.0 * df.G)]

    # jokic
    dfj = df[df.Player == 'Nikola Jokić']

    # box plot PER
    draw_one_boxplot(
        df,
        dfj,
        'PER',
        ['PER'],
        'PER'
    )

    # box plot WS/48
    draw_one_boxplot(
        df,
        dfj,
        'WS/48',
        ['WS/48'],
        'WS per 48'
    )

    # box plot OBPM, DBPM, BPM
    draw_three_boxplots(
        df,
        dfj,
        ['OBPM', 'DBPM', 'BPM'],
        ['OBPM', 'DBPM', 'BPM'],
        'Box Plus Minus'
    )

    # box plot VORP
    draw_one_boxplot(
        df,
        dfj,
        'VORP',
        ['VORP'],
        'VORP'
    )


if __name__ == "__main__":
    title1 = 'Jokić\'s traditional stats'
    jokic_stats_plot(Path('../data/per_game_data.csv'), ['Age', 'PTS', 'AST', 'TOV', 'TRB'], title1)

    title2 = 'Jokić\'s advanced stats'
    jokic_stats_plot(Path('../data/advanced_data.csv'), ['Age', 'PER', 'WS', 'BPM', 'VORP'], title2)

    compare_jokic_to_centers_traditional()
    compare_jokic_to_centers_advanced()
