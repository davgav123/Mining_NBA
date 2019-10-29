#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path
import pandas as pd
from numpy import around


if __name__ == "__main__":
    # harden PPG is from 2018-19 season
    # Bryant PPG is from 2005-06 season
    per_game_df = pd.read_csv(Path('../data/harden_and_bryant_per_game_data.csv'))
    per_48_df = pd.read_csv(Path('../data/harden_and_bryant_per_48_data.csv'))
    per_100_df = pd.read_csv(Path('../data/harden_and_bryant_per_100_data.csv'))

    avg_TS_for_2018_19_season = 0.560 # source: https://www.basketball-reference.com/leagues/NBA_2019.html#all_misc_stats
    avg_TS_for_2005_06_season = 0.536 # source: https://www.basketball-reference.com/leagues/NBA_2006.html#all_misc_stats
    avg_TS_for_1986_87_season = 0.538 # source: https://www.basketball-reference.com/leagues/NBA_1987.html#all_misc_stats

    # per game
    per_game_harden = per_game_df[per_game_df['Player'] == 'James Harden']
    per_game_bryant = per_game_df[per_game_df['Player'] == 'Kobe Bryant']
    per_game_jordan = per_game_df[per_game_df['Player'] == 'Michael Jordan']

    harden_ppg = per_game_harden['PTS'].values[0]
    bryant_ppg = per_game_bryant['PTS'].values[0]
    jordan_ppg = per_game_jordan['PTS'].values[0]

    # shooting stats
    harden_efg = per_game_harden['eFG%'].values[0]
    bryant_efg = per_game_bryant['eFG%'].values[0]
    jordan_efg = per_game_jordan['eFG%'].values[0]
    harden_ts = per_game_harden['TS%'].values[0]
    bryant_ts = per_game_bryant['TS%'].values[0]
    jordan_ts = per_game_jordan['TS%'].values[0]

    # number of games
    harden_g = per_game_harden['G'].values[0]
    bryant_g = per_game_bryant['G'].values[0]
    jordan_g = per_game_jordan['G'].values[0]

    # minutes per game
    harden_mpg = per_game_harden['MP'].values[0]
    bryant_mpg = per_game_bryant['MP'].values[0]
    jordan_mpg = per_game_jordan['MP'].values[0]

    # per 48
    per_48_harden = per_48_df[per_48_df['Player'] == 'James Harden']
    per_48_bryant = per_48_df[per_48_df['Player'] == 'Kobe Bryant']
    per_48_jordan = per_48_df[per_48_df['Player'] == 'Michael Jordan']

    harden_pp48 = per_48_harden['PTS'].values[0]
    bryant_pp48 = per_48_bryant['PTS'].values[0]
    jordan_pp48 = per_48_jordan['PTS'].values[0]

    # per 100
    per_100_harden = per_100_df[per_100_df['Player'] == 'James Harden']
    per_100_bryant = per_100_df[per_100_df['Player'] == 'Kobe Bryant']
    per_100_jordan = per_100_df[per_100_df['Player'] == 'Michael Jordan']

    harden_pp100 = per_100_harden['PTS'].values[0]
    bryant_pp100 = per_100_bryant['PTS'].values[0]
    jordan_pp100 = per_100_jordan['PTS'].values[0]

    print('James Harden in 2018-19: {} games, {} PPG, {}eFG%, {}TS% in {} minutes per game'
                    .format(harden_g, harden_ppg, harden_efg, harden_ts, harden_mpg))

    print('He was {} more efficient than the average player in was that season'
                    .format(around(harden_ts - avg_TS_for_2018_19_season, 3)))

    print('In the same season, he had {} Points per 48 minutes, and {} Points per 100 possessions'
                    .format(harden_pp48, harden_pp100))

    print('\n------------------------------------------------------------------------------------------\n')
    print('Kobe Bryant in 2005-06: {} games, {} PPG, {}eFG%, {}TS% in {} minutes per game'
                    .format(bryant_g, bryant_ppg, bryant_efg, bryant_ts, bryant_mpg))

    print('He was {} more efficient than the average player was in that season'
                    .format(around(bryant_ts - avg_TS_for_2005_06_season, 3)))

    print('In the same season, he had {} Points per 48 minutes, and {} Points per 100 possessions'
                    .format(bryant_pp48, bryant_pp100))

    print('\n------------------------------------------------------------------------------------------\n')
    print('Michael Jordan in 1986-87: {} games, {} PPG, {}eFG%, {}TS% in {} minutes per game'
                    .format(jordan_g, jordan_ppg, jordan_efg, jordan_ts, jordan_mpg))

    print('He was {} more efficient than the average player was in that season'
                    .format(around(jordan_ts - avg_TS_for_1986_87_season, 3)))

    print('In the same season, he had {} Points per 48 minutes, and {} Points per 100 possessions'
                    .format(jordan_pp48, jordan_pp100))
    