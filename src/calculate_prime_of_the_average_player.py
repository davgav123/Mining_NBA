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


def plot_age_histogram_for_every_season(mpg_limit, gms_limit):
    df_totals = pd.read_csv(Path('../data/totals_data.csv'))
    
    df_totals = filter_data_by_mpg_and_gms(df_totals, mpg_limit, gms_limit)
    ages = df_totals['Age'].values
    
    plt.hist(ages, ec='black', facecolor='blue', alpha=0.5)
    plt.title('Age histogram', fontsize=22)
    plt.xlabel('Age', fontsize='14')
    plt.show()


def list_of_frequencies(l):
    sorted_l = sorted(l)
    unique_elements = np.unique(sorted_l)

    freq_of_the_sorted = []
    for elem in unique_elements:
        freq_of_the_sorted += [(sorted_l == elem).sum()]

    # frequence of the i-th element in unique_list
    # is on the i-th spot in the frequence_list
    return freq_of_the_sorted, unique_elements


def plot_award_ages():
    mvp_df = pd.read_csv(Path('../data/reg_season_mvp.csv'))
    fmvp_df = pd.read_csv(Path('../data/finals_mvp.csv'))
    dpoy_df = pd.read_csv(Path('../data/reg_season_dpoy.csv'))
    smoy_df = pd.read_csv(Path('../data/sixth_man.csv'))

    mvp_ages = mvp_df['Age'].values
    fmvp_ages = fmvp_df['Age'].values
    dpoy_ages = dpoy_df['Age'].values
    smoy_ages = smoy_df['Age'].values

    mvp_age_freq, mvp_ages = list_of_frequencies(mvp_ages)
    fmvp_age_freq, fmvp_ages = list_of_frequencies(fmvp_ages)
    dpoy_age_freq, dpoy_ages = list_of_frequencies(dpoy_ages)
    smoy_age_freq, smoy_ages = list_of_frequencies(smoy_ages)

    plt.subplot(2, 2, 1)
    plt.bar(mvp_ages, mvp_age_freq)
    x = np.arange(min(mvp_ages), max(mvp_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Number of MVP awards', fontsize=12)
    plt.title('Bar plot of the age of the players who won MVP', fontsize=16)

    plt.subplot(2, 2, 2)
    plt.bar(fmvp_ages, fmvp_age_freq)
    x = np.arange(min(fmvp_ages), max(fmvp_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Number of FMVP awards', fontsize=12)
    plt.title('Bar plot of the age of the players who won FMVP', fontsize=16)

    plt.subplot(2, 2, 3)
    plt.bar(dpoy_ages, dpoy_age_freq)
    x = np.arange(min(dpoy_ages), max(dpoy_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Number of DPOY awards', fontsize=12)
    plt.title('Bar plot of the age of the players who won DPOY', fontsize=16)

    plt.subplot(2, 2, 4)
    plt.bar(smoy_ages, smoy_age_freq)
    x = np.arange(min(smoy_ages), max(smoy_ages) + 1)
    plt.xticks(x, x)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Number of SMOY awards', fontsize=12)
    plt.title('Bar plot of the age of the players who won SMOY', fontsize=16)

    plt.show()


if __name__ == "__main__":
    plot_age_histogram_for_every_season(15, 35)
    plot_award_ages()
