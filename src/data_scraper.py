#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# from 1979-1980 to 2018-2019 season, 3-point era
three_point_era_years = [year for year in range(1980, 2020)]

per_game_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_game.html'
totals_url = 'https://www.basketball-reference.com/leagues/NBA_{}_totals.html'
per_36_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_minute.html'
per_100_poss_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_poss.html'
advanced_url = 'https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'


# source for this code: https://towardsdatascience.com/web-scraping-nba-stats-4b4f8c525994
def scrap_bbref_table(year, target_table_url):
    url = target_table_url.format(year)
    
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    # get headers
    headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    headers = headers[1:]

    # rest of the data
    rows = soup.findAll('tr')[1:]
    player_stats = [[td.getText() for td in rows[i].findAll('td')]
                    for i in range(len(rows))]

    stats = pd.DataFrame(player_stats, columns=headers)
    
    # delete empty rows, rows where 'Player' field is missing
    # those rows are missing because of html
    stats = stats[pd.notnull(stats['Player'])]
    stats = stats.reset_index(drop=True)

    print('Reading data from {} finished'.format(url))
    return stats


def data_from_url(url, dest_path, years):
    dest_df = pd.DataFrame()

    for year in years:
        df = scrap_bbref_table(year, url)

        # add season colum
        df['season'] = year

        dest_df = pd.concat([dest_df, df], ignore_index=True)

    dest_df.to_csv(dest_path, index=False)
    print('Data is saved into {}'.format(dest_path))


if __name__ == "__main__":
    data_from_url(per_game_url, Path('../data/per_game_data.csv'), three_point_era_years)
    data_from_url(totals_url, Path('../data/totals_data.csv'), three_point_era_years)
    data_from_url(per_36_url, Path('../data/per_36_data.csv'), three_point_era_years)
    data_from_url(per_100_poss_url, Path('../data/per_100_data.csv'), three_point_era_years)
    data_from_url(advanced_url, Path('../data/advanced_data.csv'), three_point_era_years)    

