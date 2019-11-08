#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from nba_api.stats.endpoints.shotchartdetail import ShotChartDetail

# function that draws the basketball court by: github.com/bradleyfay
# source: https://github.com/bradleyfay/py-Goldsberry/blob/master/docs/Visualizing%20NBA%20Shots%20with%20py-Goldsberry.ipynb
from matplotlib.patches import Circle, Rectangle, Arc
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


def plot_raw_shotchart(data_frame, title, point_color):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(12, 12))

    paths = ax.scatter(
        x=data_frame.LOC_X,
        y=data_frame.LOC_Y,
        marker='o',
        c=point_color,
        s=30,
        alpha=0.6
    )

    # Remove ticks
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.grid(False)

    plt.title(title, fontsize=40)

    # Draw court
    draw_court(ax=ax,outer_lines=True, lw=3)
    ax.set_xlim(-251,251)
    ax.set_ylim(-65,423)


def team_shotchart(team_id, first_date, last_date, title, point_color):
    # scrap shots by selected team from stats.nba.com
    team_shots = ShotChartDetail(
        team_id=team_id, 
        player_id=0, 
        context_measure_simple='FGA'
    )
    team_shots = team_shots.get_data_frames()[0]

    # filter games
    team_shots = team_shots[team_shots.GAME_DATE > first_date]
    team_shots = team_shots[team_shots.GAME_DATE < last_date]

    # draw shotchart
    plot_raw_shotchart(team_shots, title, point_color)
    plt.show()


if __name__ == "__main__":
    team_id_color = [
        ['HOU', 1610612745, 'red'],
        ['MLK', 1610612749, '#004d00'],
        ['DEN', 1610612743, '#0066cc'],
        ['BRK', 1610612751, 'black']
    ]

    # dates used for filtering must be type string!
    # date format: yyyymmdd
    seasons = [['20181001', '20191001'], ['20121001', '20131001']]

    # draw shotchart for every team in selected seasons
    for s in seasons:
        for tic in team_id_color:
            team_shotchart(
                tic[1], # id
                s[0],
                s[1],
                f'{tic[0]} shotchart for {s[0][:4]}-{s[1][2:4]} season',
                tic[2] # color
            )
