#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def filter_data_by_mpg_and_gms(df, mpg_limit, gms_limit):
    '''
        Does not work for per_game_data (mpg filter)
    '''
    df = df[df.MP > (mpg_limit * df.G)] 
    df = df[df.G > gms_limit] 

    return df


def plot_age_histogram(filter_players=True, mpg_limit=15.0, gms_limit=35):
    df_totals = pd.read_csv(Path('../data/totals_data.csv'))
    
    if filter_players:
        df_totals = filter_data_by_mpg_and_gms(df_totals, mpg_limit, gms_limit)
    
    ages = df_totals['Age'].values
    
    plt.hist(ages, ec='black', facecolor='blue', alpha=0.5)
    plt.title('Age histogram', fontsize=22)
    plt.xlabel('Age', fontsize='14')
    plt.show()


def plot_award_ages():
    mvp_df = pd.read_csv(Path('../data/reg_season_mvp.csv'))
    fmvp_df = pd.read_csv(Path('../data/finals_mvp.csv'))
    dpoy_df = pd.read_csv(Path('../data/reg_season_dpoy.csv'))
    smoy_df = pd.read_csv(Path('../data/sixth_man.csv'))

    mvp_df = mvp_df.groupby('Age').count().reset_index()
    fmvp_df = fmvp_df.groupby('Age').count().reset_index()
    dpoy_df = dpoy_df.groupby('Age').count().reset_index()
    smoy_df = smoy_df.groupby('Age').count().reset_index()

    mvp_ages = mvp_df['Age']
    mvp_age_freq = mvp_df['Player'] # could be any stat but age

    # box plot for mvp
    plt.subplot(2, 2, 1)
    plt.bar(mvp_ages, mvp_age_freq)
    x = np.arange(min(mvp_ages), max(mvp_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of MVP awards', fontsize=14)
    plt.title('Bar plot of the age of the players who won MVP', fontsize=16)

    fmvp_ages = fmvp_df['Age']
    fmvp_age_freq = fmvp_df['Player'] # could be any stat but age

    # box plot for fmvp    
    plt.subplot(2, 2, 2)
    plt.bar(fmvp_ages, fmvp_age_freq)
    x = np.arange(min(fmvp_ages), max(fmvp_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of FMVP awards', fontsize=14)
    plt.title('Bar plot of the age of the players who won FMVP', fontsize=16)

    dpoy_ages = dpoy_df['Age']
    dpoy_age_freq = dpoy_df['Player'] # could be any stat but age

    # box plot for dpoy
    plt.subplot(2, 2, 3)
    plt.bar(dpoy_ages, dpoy_age_freq)
    x = np.arange(min(dpoy_ages), max(dpoy_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of DPOY awards', fontsize=14)
    plt.title('Bar plot of the age of the players who won DPOY', fontsize=16)

    smoy_ages = smoy_df['Age']
    smoy_age_freq = smoy_df['Player'] # could be any stat but age

    # box plot for smoy
    plt.subplot(2, 2, 4)
    plt.bar(smoy_ages, smoy_age_freq)
    x = np.arange(min(smoy_ages), max(smoy_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of SMOY awards', fontsize=14)
    plt.title('Bar plot of the age of the players who won SMOY', fontsize=16)

    plt.show()


def bar_plot_stat_by_age(path_to_csv, stat_of_interest, filter_players=False, mpg=15.0, gms=35):
    df = pd.read_csv(Path(path_to_csv))

    if filter_players:
        df = filter_data_by_mpg_and_gms(df, mpg, gms)

    # average selected stat by age
    df = df.groupby('Age')[stat_of_interest].mean().reset_index()
    
    ages = df['Age']
    stat = df[stat_of_interest]
    
    plt.bar(ages, stat)
    x = np.arange(min(ages), max(ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('Average number of {} per age by a player per season'.format(stat_of_interest),
                fontsize=16)
    plt.show()


def player_stat_by_age(name, path_to_csv, stat):
    df_per_game = pd.read_csv(Path(path_to_csv))
    df = df_per_game[df_per_game['Player'] == name]

    # remove '*' from name, if there is one
    name = name[:-1] if name[-1] == '*' else name

    age = df['Age']
    stats = df[stat]

    plt.bar(age, stats)
    x = np.arange(min(age), max(age) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('{}\'s {} per season'.format(name, stat), fontsize=16)
    plt.show()


if __name__ == "__main__":
    plot_age_histogram(True, 15, 35)
    plot_award_ages()
    bar_plot_stat_by_age('../data/totals_data.csv', 'PTS')
    bar_plot_stat_by_age('../data/advanced_data.csv', 'BPM')

    # '*' represents hall of famer
    player_stat_by_age('Shaquille O\'Neal*', '../data/advanced_data.csv', 'WS/48')
    player_stat_by_age('Tim Duncan', '../data/advanced_data.csv', 'WS/48')