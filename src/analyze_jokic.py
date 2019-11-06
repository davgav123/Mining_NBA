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


def draw_one_boxplot(df, df_jokic, stat, lbls, title, show_plot=False):
    plt.boxplot(df[stat], notch=True, showfliers=False)
    plt.scatter([1 for i in range(len(df[stat]))], df[stat], c='red')
    jokic = plt.scatter([1], df_jokic[stat], c='blue', s=90)

    plt.xticks([1], labels=lbls, fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid(axis='y')
    plt.legend(handles=[jokic], labels=['Jokić'], fontsize=22)
    plt.title(title, fontsize=36)

    if show_plot:
        plt.show()


def draw_two_boxplots(df, df_jokic, stats, lbls, title, show_plot=False):
    plt.boxplot([df[stats[0]], df[stats[1]]], notch=True, showfliers=False)
    plt.scatter([1 for i in range(len(df[stats[0]]))], df[stats[0]], c='red')
    plt.scatter([2 for i in range(len(df[stats[1]]))], df[stats[1]], c='red')
    jokic = plt.scatter([1, 2], [df_jokic[stats[0]].values, df_jokic[stats[1]].values], c='blue', s=90)

    plt.xticks([1, 2], labels=lbls, fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid(axis='y')
    plt.legend(handles=[jokic], labels=['Jokić'], fontsize=22)
    plt.title(title, fontsize=36)

    if show_plot:
        plt.show()


def draw_three_boxplots(df, df_jokic, stats, lbls, title, show_plot=False):
    plt.boxplot([df[stats[0]], df[stats[1]], df[stats[2]]], notch=True, showfliers=False)
    plt.scatter([1 for i in range(len(df[stats[0]]))], df[stats[0]], c='red')
    plt.scatter([2 for i in range(len(df[stats[1]]))], df[stats[1]], c='red')
    plt.scatter([3 for i in range(len(df[stats[2]]))], df[stats[2]], c='red')
    jokic = plt.scatter([1, 2, 3], [df_jokic[stats[0]].values, df_jokic[stats[1]].values, df_jokic[stats[2]].values], c='blue', s=90)

    plt.xticks([1, 2, 3], labels=lbls, fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid(axis='y')
    plt.legend(handles=[jokic], labels=['Jokić'], fontsize=22)
    plt.title(title, fontsize=36)

    if show_plot:
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
    plt.subplot(1, 3, 1)
    draw_one_boxplot(
        df,
        dfj,
        'PTS',
        ['PTS/G'],
        'PTS/G'
    )

    # box plot DRB, ORB and TRB
    plt.subplot(1, 3, 2)
    draw_three_boxplots(
        df,
        dfj,
        ['DRB', 'ORB', 'TRB'],
        ['DRB/G', 'ORB/G', 'TRB/G'],
        'Rebounds/G'
    )

    # box plot AST and TOV
    plt.subplot(1, 3, 3)
    draw_two_boxplots(
        df,
        dfj,
        ['AST', 'TOV'],
        ['AST/G', 'TOV/G'],
        'AST/G and TOV/G',
        show_plot=True
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
    plt.subplot(1, 3, 1)
    draw_one_boxplot(
        df,
        dfj,
        'WS/48',
        ['WS/48'],
        'WS per 48'
    )

    # box plot WS/48
    plt.subplot(1, 3, 2)
    draw_one_boxplot(
        df,
        dfj,
        'PER',
        ['PER'],
        'PER'
    )

    # box plot VORP
    plt.subplot(1, 3, 3)
    draw_one_boxplot(
        df,
        dfj,
        'VORP',
        ['VORP'],
        'VORP',
        show_plot=True
    )

    # box plot OBPM, DBPM, BPM
    draw_three_boxplots(
        df,
        dfj,
        ['OBPM', 'DBPM', 'BPM'],
        ['OBPM', 'DBPM', 'BPM'],
        'Box Plus Minus',
        show_plot=True
    )


# after this function plt.show() should be called
def scatter_names(df, x_col, y_col, players, name_x, name_y):
    for player in players:
        dfp = df[df.Player == player]
        plt.text(x=dfp[x_col] + name_x, y=dfp[y_col] + name_y, s=player, c='black', fontsize=15)


def jokic_and_players_2018_19_season(data_path, x, y, name_x, name_y, advanced=False):
    df = pd.read_csv(Path(data_path))

    df = df[df.season == 2019]

    # players that are contributors
    df = df[df.G > 35]
    if not advanced:
        df = df[df.MP > 15.0]
    else:
        df = df[df.MP > (15.0 * df.G)]

    # jokic
    dfj = df[df.Player == 'Nikola Jokić']

    # Point guards
    dfpg = df[df.Pos.isin(['PG', 'PG-SG'])]

    # every player in a season plot
    plt.scatter(df[x], df[y], c='red', s=80)
    # point guard plot
    pg = plt.scatter(dfpg[x], dfpg[y], c='green', s=80)
    # jokic plot
    nj = plt.scatter(dfj[x], dfj[y], c='blue', s=220)

    scatter_names(
        df,
        x,
        y,
        [
            'Chris Paul',
            'LeBron James',
            'James Harden',
            'Ben Simmons',
            'Kyrie Irving',
            'Jrue Holiday',
            'Kyle Lowry',
            'Russell Westbrook'
        ],
        name_x,
        name_y
    )

    plt.grid()
    plt.legend(handles=[pg, nj], labels=['Point Guards', 'Nikola Jokić'], fontsize=22, loc=2)
    plt.title('Players from 2018-19 season', fontsize=36)

    plt.xlabel('AST%' if advanced else 'AST/G', fontsize=24)
    plt.ylabel('TOV%' if advanced else 'TOV/G', fontsize=24)
    plt.show()


def jokic_and_players(data_path, x, y, advanced=False):
    df = pd.read_csv(Path(data_path))

    # players that are contributors
    df = df[df.G > 35]
    if not advanced:
        df = df[df.MP > 15.0]
    else:
        df = df[df.MP > (15.0 * df.G)]

    # jokic
    dfj = df[df.Player == 'Nikola Jokić']
    dfj = dfj[dfj.season == 2019]

    # Point guards
    dfpg = df[df.Pos.isin(['PG', 'PG-SG'])]

    # Centers
    dfc = df[df.Pos.isin(['C', 'C-PF'])]

    # every player in a season plot
    plt.scatter(df[x], df[y], c='red', s=80)
    # point guard plot
    pg = plt.scatter(dfpg[x], dfpg[y], c='orange', s=80)
    # centers plot
    c = plt.scatter(dfc[x], dfc[y], c='purple', s=80)
    # jokic plot
    nj = plt.scatter(dfj[x], dfj[y], c='blue', s=220)

    plt.grid()
    plt.legend(handles=[pg, c, nj], labels=['Point Guards', 'Centers', 'Nikola Jokić 2018-19'], fontsize=22, loc=2)
    plt.title('Players from 3-point era', fontsize=36)

    plt.xlabel('AST%' if advanced else 'AST/G', fontsize=24)
    plt.ylabel('TOV%' if advanced else 'TOV/G', fontsize=24)
    plt.show()


if __name__ == "__main__":
    title1 = 'Jokić\'s traditional stats'
    jokic_stats_plot(Path('../data/per_game_data.csv'), ['Age', 'PTS', 'AST', 'TOV', 'TRB'], title1)

    title2 = 'Jokić\'s advanced stats'
    jokic_stats_plot(Path('../data/advanced_data.csv'), ['Age', 'PER', 'WS', 'BPM', 'VORP'], title2)

    compare_jokic_to_centers_traditional()
    compare_jokic_to_centers_advanced()

    jokic_and_players_2018_19_season('../data/per_game_data.csv', 'AST', 'TOV', 0.05, -0.15, advanced=False)
    jokic_and_players_2018_19_season('../data/advanced_data.csv', 'AST%', 'TOV%', 0.2, -0.7, advanced=True)

    jokic_and_players('../data/per_game_data.csv', 'AST', 'TOV', advanced=False)
    jokic_and_players('../data/advanced_data.csv', 'AST%', 'TOV%', advanced=True)
