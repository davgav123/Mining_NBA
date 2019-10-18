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


def data_from_url(url, dest_path):
    dest_df = pd.DataFrame()

    for year in three_point_era_years:
        df = scrap_bbref_table(year, url)
        dest_df = pd.concat([dest_df, df], ignore_index=True)

    dest_df.to_csv(dest_path)
    print('Data is saved into {}'.format(dest_path))


if __name__ == "__main__":
    data_from_url(per_game_url, '../data/per_game_data.csv')
    data_from_url(totals_url, '../data/totals_data.csv')
    data_from_url(per_36_url, '../data/per_36_data.csv')
    data_from_url(per_100_poss_url, '../data/per_100_data.csv')
    data_from_url(advanced_url, '../data/advanced_data.csv')    

