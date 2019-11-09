#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints.shotchartdetail import ShotChartDetail
import matplotlib.pyplot as plt
import numpy as np


def scrap_shooting_data(ids, path_to_dest, from_season='20181001', to_season='20191001'):
    df = pd.DataFrame()

    for id_i in team_ids:
        team_shots = ShotChartDetail(
            team_id=id_i, 
            player_id=0, 
            context_measure_simple='FGA'
        )

        # get data frames
        ts_df = team_shots.get_data_frames()[0]
        
        # filter only 2018-19 season
        ts_df = ts_df[ts_df.GAME_DATE > from_season]
        ts_df = ts_df[ts_df.GAME_DATE < to_season]

        # add ts_df to df
        df = pd.concat([df, ts_df], ignore_index=True)

    # save
    df.to_csv(Path(path_to_dest))
    print('Data saved to {}'.format(path_to_dest))


def shots_by_area(df, zone_range):
    ''' calculates number of shots taken from zone_range parameter '''
    return df[df['SHOT_ZONE_RANGE'] == zone_range]['SHOT_MADE_FLAG'].values[0]


def shot_percentage_by_zone_range_2018_19(print_result=True):
    df = pd.read_csv(Path('../data/shooting_2018_19_data.csv'))

    df = df[['SHOT_ZONE_RANGE', 'SHOT_MADE_FLAG']]
    
    # made shots
    df_made = df[df['SHOT_MADE_FLAG'] == 1]    
    df_made = df_made.groupby('SHOT_ZONE_RANGE').count().reset_index()

    # attempted shots
    df_attempted = df.groupby('SHOT_ZONE_RANGE').count().reset_index()

    # calculate percentage by zone and add free throw percentage
    shoot_pct_by_zone = {
        '16-24 ft.': shots_by_area(df_made, '16-24 ft.') / shots_by_area(df_attempted, '16-24 ft.'),
        '24+ ft.': shots_by_area(df_made, '24+ ft.') / shots_by_area(df_attempted, '24+ ft.'),
        '8-16 ft.': shots_by_area(df_made, '8-16 ft.') / shots_by_area(df_attempted, '8-16 ft.'),
        'Back Court Shot': shots_by_area(df_made, 'Back Court Shot') / shots_by_area(df_attempted, 'Back Court Shot'),
        'Less Than 8 ft.': shots_by_area(df_made, 'Less Than 8 ft.') / shots_by_area(df_attempted, 'Less Than 8 ft.'),
        'Free Throw Shot': 0.766
    }

    if print_result:
        for k, v in shoot_pct_by_zone.items():
            print(k, v)

    return shoot_pct_by_zone


def percentage_of_mid_range_shots():
    df = pd.read_csv(Path('../data/shooting_2018_19_data.csv'))

    df = df[['TEAM_NAME', 'SHOT_ZONE_BASIC']]

    # take mid-range shots and count them by team
    df_mid_range = df[df['SHOT_ZONE_BASIC'] == 'Mid-Range']
    df_mid_range_by_team = df_mid_range.groupby('TEAM_NAME').count().reset_index()

    # every shot (including mid-range) count by team
    df_shots_by_team = df.groupby('TEAM_NAME').count().reset_index()

    # create new dataframe with information about percent of the shots from mid-range for every team
    # that dataframe will have two columns [team_name, pct_of_mid_range], and it is used to sort by team name
    df_pct_mid_range = pd.DataFrame(columns=['TEAM_NAME', 'PCT_FROM_MID_RANGE'])

    for team in df_mid_range_by_team['TEAM_NAME'].values:
        # maybe looks ugly but otherwise the line would be to long
        pct = float(df_mid_range_by_team[df_mid_range_by_team['TEAM_NAME'] == team]['SHOT_ZONE_BASIC'].values[0]) \
            / float(df_shots_by_team[df_shots_by_team['TEAM_NAME'] == team]['SHOT_ZONE_BASIC'].values[0])

        # add pair (team, and pct of mid-range for that team) into dataframe
        df_pct_mid_range = df_pct_mid_range.append(pd.DataFrame([[team, pct]], columns=['TEAM_NAME', 'PCT_FROM_MID_RANGE']))

    # Atlanta first, Washington last
    df_pct_mid_range = df_pct_mid_range.sort_values(by=['TEAM_NAME'])

    return df_pct_mid_range


def compare_ortg_and_mid_range_pct():
    df_ortg = pd.read_csv(Path('../data/teams_season_2018_19_per_100.csv'))[['Team', 'PTS']]
    df_pct_mid_range = percentage_of_mid_range_shots()

    # this are the columns of need
    pct = df_pct_mid_range['PCT_FROM_MID_RANGE'].values
    ortg = df_ortg['PTS'].values

    plt.scatter(x=pct, y=ortg, s=80)

    # add team name for every dot
    for index, row in df_ortg.iterrows():
        plt.text(x=pct[index], y=row['PTS'] + 0.1, s=row['Team'], fontsize=16)

    # add line that represents average ORtg
    avg = plt.axhline(y=110.4, color='red')

    plt.grid()
    plt.legend(handles=[avg], labels=['Average ORtg, 110.4'], fontsize=22, loc=4)

    plt.xticks(
        ticks=np.arange(0.0, 0.31, 0.05),
        labels=[str(i) + '%' for i in range(0, 31, 5)],
        fontsize=14
    )
    plt.yticks(fontsize=14)

    plt.xlabel('% of shots that were mid-range shots', fontsize=24)
    plt.ylabel('ORtg', fontsize=24)
    plt.show()


def print_correlations():
    '''
    this function prints correlations
    between ORtg, TS%, %of mid-range attempts
    and % of 3-point attempts from data
    from 2000-01 to 2018-19 seasons
    '''

    df = pd.read_csv(Path('../data/ts_ortg_by_season_data.csv'))

    print('Correlation between TS% and ORtg: {}\n'.format(df['ORtg'].corr(df['TS%'])))
    print('Correlation between TS% and % of 3PA: {}\n'.format(df['TS%'].corr(df['PCT_3PA'])))
    print('Correlation between TS% and % of MidRangeAttempts: {}\n'.format(df['TS%'].corr(df['PCT_MidRangeA'])))
    print('Correlation between % of 3PA and ORtg: {}\n'.format(df['PCT_3PA'].corr(df['ORtg'])))
    print('Correlation between % of MidRangeAttempts and ORtg: {}\n'.format(df['PCT_MidRangeA'].corr(df['ORtg'])))


if __name__ == "__main__":
    team_ids = list(range(1610612737, 1610612767))
    # scrap_shooting_data(team_ids, '../data/shooting_2018_19_data.csv')

    shot_percentage_by_zone_range_2018_19()

    compare_ortg_and_mid_range_pct()

    print_correlations()
