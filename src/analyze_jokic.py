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

    plt.title(title, fontsize=36)    
    plt.legend(stats, loc=0, fontsize=22)
    plt.show()


if __name__ == "__main__":
    title1 = 'Jokić\'s traditional stats'
    jokic_stats_plot(Path('../data/per_game_data.csv'), ['Age', 'PTS', 'AST', 'TOV', 'TRB'], title1)
    
    title2 = 'Jokić\'s advanced stats'
    jokic_stats_plot(Path('../data/advanced_data.csv'), ['Age', 'PER', 'WS', 'BPM', 'VORP'], title2)