from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

# from 1979-1980 to 2018-2019 seasons, 3-point era
three_point_era_years = [year for year in range(1980, 2020)]

per_game_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_game.html'
totals_url = 'https://www.basketball-reference.com/leagues/NBA_{}_totals.html'
per_36_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_minute.html'
per_100_poss_url = 'https://www.basketball-reference.com/leagues/NBA_{}_per_poss.html'
advanced_url = 'https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'


def scrap_bbref_table(year, target_table_url):
    url = target_table_url.format(year)
    
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    headers = headers[1:]

    rows = soup.findAll('tr')[1:]
    player_stats = [[td.getText() for td in rows[i].findAll('td')]
                    for i in range(len(rows))]

    stats = pd.DataFrame(player_stats, columns = headers)

    print('Reading data from {} finished'.format(url))
    return stats


def create_per_game_dataFrame():
    per_game_df = pd.DataFrame()

    for year in three_point_era_years:
        df, col = scrap_bbref_table(year, per_game_url)
        per_game_df = pd.concat([per_game_df, df], ignore_index=True)

    per_game_df.to_csv('../data/per_game_data.csv')
    print('Per game data created!')


def create_totals_dataFrame():
    totals_df = pd.DataFrame()

    for year in three_point_era_years:
        df = scrap_bbref_table(year, totals_url)
        totals_df = pd.concat([totals_df, df], ignore_index=True)

    totals_df.to_csv('../data/totals_data.csv')
    print('Totals data created!')


def create_per_36_dataFrame():
    per_36_df = pd.DataFrame()

    for year in three_point_era_years:
        df = scrap_bbref_table(year, per_36_url)
        per_36_df = pd.concat([per_36_df, df], ignore_index=True)

    per_36_df.to_csv('../data/per_36_data.csv')
    print('Per 36 data created!')


def create_per_100_poss_dataFrame():
    per_100_poss_df = pd.DataFrame()

    for year in three_point_era_years:
        df = scrap_bbref_table(year, per_100_poss_url)
        per_100_poss_df = pd.concat([per_100_poss_df, df], ignore_index=True)

    per_100_poss_df.to_csv('../data/per_100_poss_data.csv')
    print('Per 100 possessions data created!')


def create_advanced_dataFrame():
    advanced_df = pd.DataFrame()

    for year in three_point_era_years:
        df = scrap_bbref_table(year, advanced_url)
        advanced_df = pd.concat([advanced_df, df], ignore_index=True)

    advanced_df.to_csv('../data/advanced_data.csv')
    print('Advanced data created!')
    

if __name__ == "__main__":
    create_per_game_dataFrame()
    

