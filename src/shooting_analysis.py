#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints.shotchartdetail import ShotChartDetail


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


if __name__ == "__main__":
    team_ids = list(range(1610612737, 1610612766))
    scrap_shooting_data(team_ids, '../data/shooting_2018_19_data.csv')

    shot_percentage_by_zone_range_2018_19()
