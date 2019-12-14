#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
from copy import copy
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def transform_positions(position):
    '''
        Positon can be PG, SG, SF, PF, C or combination of the two,
        for example PF-C or SF-SG. If that is the case, only first 
        position is taken because player played more on that one, 
        and we want to stay consistent (one player one position).
    '''

    if len(position) > 2:
        position = position[:position.index('-')]

    return position


def filter_data(df):
    # filter seasons
    df = df[df.season > 1991] # from 1990-91 season
    df = df[df.season < 2011] # t0 2010-11 season

    # filter contributors
    df = df[df.MP >= 15.0]
    df = df[df.G > 35]

    # filter needed columns
    df.drop(columns=['Tm', 'Age', 'G', 'GS', 'MP', 'Player', 'season'], inplace=True)

    # drop percentage columns (can contain NaN values)
    df.drop(columns=['FG%', '3P%', '2P%', 'FT%', 'eFG%'], inplace=True)

    # filter position (if pos is  for example 'SG-PG' we want just 'SG')
    df['Pos'] = df['Pos'].apply(transform_positions)

    return df


def best_classifier(model, grid, model_name, X_train, X_test, y_train, y_test):    
    clf = GridSearchCV(model, param_grid=grid, scoring='accuracy')
    clf.fit(X_train, y_train)

    print(f'Best params for {model_name}: {clf.best_params_}')
    print('') # new line


def run_best_classifiers(X_train, X_test, y_train, y_test):
    '''
        Run different classification algorithms
        with different parameters. Grid search is used.
        Best models are then created separately. 
    '''

    best_classifier(
        KNeighborsClassifier(),
        {
            'n_neighbors': [2, 5, 7, 10, 12, 15], 
            'weights': ['uniform', 'distance']
        },
        'KNN',
        X_train, X_test, y_train, y_test
    )

    best_classifier(
        RandomForestClassifier(),
        {
            'n_estimators': np.linspace(start=50, stop=250, num=5, dtype=int), 
            'criterion': ['gini', 'entropy'], 
            'max_depth': np.linspace(start=10, stop=100, num=10, dtype=int),
            'random_state': [27]
        },
        'Random forest',
        X_train, X_test, y_train, y_test
    )

    best_classifier(
        SVC(),
        {
            'C': [0.1, 1, 10, 100], 
            'kernel': ['rbf', 'linear'], 
            'gamma': ['scale', 0.01, 0.00001]
        },
        'SVM',
        X_train, X_test, y_train, y_test
    )


def create_classifier(model, model_name, X_train, X_test, y_train, y_test):
    clf = model
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'{model_name} train accuracy: {clf.score(X_train, y_train)}')
    print(f'{model_name} test accuracy: {clf.score(X_test, y_test)}')
    print(f'{model_name} classification report: \n{classification_report(y_test, y_pred)}')
    print(f'{model_name} confusion matrix: \n{confusion_matrix(y_test, y_pred)}')

    print('') # new line

    return clf


def classify_selected_player(per_game_data, player_name, trained_classifier):
    player = copy(per_game_data[per_game_data.Player == player_name])
    seasons = player['season'].values
    positions = player['Pos'].values

    # prepare data from classification
    player.drop(columns=['Tm', 'Pos', 'Age', 'G', 'GS', 'MP', 'Player', 'season'], inplace=True)
    player.drop(columns=['FG%', '3P%', '2P%', 'FT%', 'eFG%'], inplace=True)

    print(player_name)
    print(f'Listed positions: {list(zip(seasons, positions))}')
    print(f'Predicted positions: {list(zip(seasons, trained_classifier.predict(player)))}')
    print('') # new line


def classify_players(trained_classifier, players=None):
    if not players:
        players = [
            'Dāvis Bertāns',
            'Nikola Jokić',
            'Russell Westbrook',
            'James Harden',
            'LeBron James',
            'Damian Lillard',
            'Draymond Green',
            'Hassan Whiteside',
            'Ben Simmons',
            'Giannis Antetokounmpo',
        ]

    per_game = pd.read_csv(Path('../data/per_game_data.csv'))
    for player in players:
        classify_selected_player(per_game, player, trained_classifier)


if __name__ == "__main__":
    df = pd.read_csv(Path('../data/per_game_data.csv'))

    df = filter_data(df)

    # split on data and target columns
    X = df.drop(columns=['Pos'])
    y = df['Pos']

    # number of players by positions
    print(f'Number of players by position is {dict(Counter(y))}\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

    # try different parameters for different classificators
    # run_best_classifiers(X_train, X_test, y_train, y_test)

    # create dummy classifier
    dummy = create_classifier(
        DummyClassifier(strategy='stratified'),
        'Dummy', X_train, X_test, y_train, y_test
    )

    # the best classifier from GridSearchCV
    knn = create_classifier(
        KNeighborsClassifier(n_neighbors=15, weights='distance'), 
        'KNN', X_train, X_test, y_train, y_test
    )

    random_forest = create_classifier(
        RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=250, random_state=27),
        'Random forest', X_train, X_test, y_train, y_test
    )

    svc = create_classifier(
        SVC(C=100, gamma='scale', kernel='rbf'), 'SVC',
        X_train, X_test, y_train, y_test
    )

    # classify some of the players that play today (2018-19 season)
    # classify_players(random_forest)
    # classify_players(svc)
