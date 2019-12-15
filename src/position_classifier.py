#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
    df = df[df.season >= 1991] # from 1990-91 season
    df = df[df.season <= 2011] # t0 2010-11 season

    # filter contributors
    df = df[df.MP >= 15.0]
    df = df[df.G >= 35]

    # filter needed columns
    df.drop(columns=['Tm', 'Age', 'G', 'GS', 'MP', 'Player', 'season'], inplace=True)

    # drop percentage columns (can contain NaN values)
    df.drop(columns=['FG%', '3P%', '2P%', 'FT%', 'eFG%'], inplace=True)

    # filter position (if pos is  for example 'SG-PG' we want just 'SG')
    df['Pos'] = df['Pos'].apply(transform_positions)

    return df


def plot_boxplot(x, y, title):
    sns.boxplot(x, y)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel('', fontsize=22)
    plt.ylabel('', fontsize=22)

    plt.title(title, fontsize=24)
    plt.show()


def visualize_data(df):
    plot_boxplot(df['Pos'], df['PTS'], 'PTS per game by position')

    plot_boxplot(df['Pos'], df['3PA'], '3PA per game by position')

    plot_boxplot(df['Pos'], df['2PA'], '2PA per game by position')

    plot_boxplot(df['Pos'], df['TRB'], 'TRB per game by position')

    plot_boxplot(df['Pos'], df['AST'], 'AST per game by position')

    plot_boxplot(df['Pos'], df['STL'], 'STL per game by position')

    plot_boxplot(df['Pos'], df['BLK'], 'BLK per game by position')


def best_classifier(model, grid, model_name, cv, X_train, X_test, y_train, y_test):
    clf = GridSearchCV(model, param_grid=grid, cv=cv, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f'Best params for {model_name}: {clf.best_params_}')
    print('') # new line


def find_best_classifiers(cv, X_train, X_test, y_train, y_test):
    '''
        Run different classification algorithms
        with different parameters. Grid search is used.
        Best models are then created separately. 
    '''

    best_classifier(
        KNeighborsClassifier(),
        {
            'n_neighbors': [2, 5, 7, 10, 12, 15, 17, 20],
            'weights': ['uniform', 'distance']
        },
        'KNN', cv,
        X_train, X_test, y_train, y_test
    )

    best_classifier(
        RandomForestClassifier(),
        {
            'n_estimators': np.linspace(start=50, stop=300, num=6, dtype=int),
            'criterion': ['gini', 'entropy'],
            'max_depth': np.linspace(start=10, stop=100, num=10, dtype=int),
            'random_state': [27]
        },
        'Random forest', cv,
        X_train, X_test, y_train, y_test
    )

    best_classifier(
        GradientBoostingClassifier(),
        {
            'loss': ['deviance'],
            'n_estimators': np.linspace(start=50, stop=300, num=6, dtype=int),
            'max_depth': np.linspace(start=10, stop=100, num=10, dtype=int),
            'max_features': [None, 'sqrt'],
            'random_state': [27]
        },
        'Gradient Boosting', cv,
        X_train, X_test, y_train, y_test
    )

    best_classifier(
        SVC(),
        {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 0.1, 0.01, 0.001, 0.0001]
        },
        'SVM', cv,
        X_train, X_test, y_train, y_test
    )


def create_classifier(model, model_name, X_train, X_test, y_train, y_test):
    clf = model
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'{model_name} train accuracy: {clf.score(X_train, y_train):.2f}')
    print(f'{model_name} test accuracy: {clf.score(X_test, y_test):.2f}')

    score = cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
    print(f'{model_name} cross-validation score: {score.mean():.2f} (+/- {(score.std() * 2):.2f})')

    print(f'{model_name} classification report: \n{classification_report(y_test, y_pred)}')
    print(f'{model_name} confusion matrix: \n{confusion_matrix(y_test, y_pred)}')

    print('') # new line

    return clf


def classify_selected_player(per_game_data, player_name, trained_classifiers):
    player = copy(per_game_data[per_game_data.Player == player_name])

    seasons = player['season'].values
    positions = player['Pos'].values

    # prepare data for classification
    player.drop(columns=['Tm', 'Pos', 'Age', 'G', 'GS', 'MP', 'Player', 'season'], inplace=True)
    player.drop(columns=['FG%', '3P%', '2P%', 'FT%', 'eFG%'], inplace=True)

    max_name_len = max([len(v) for (k, v) in trained_classifiers]) # for output formating

    print(player_name)
    spaces = ' ' * max_name_len # for output formatting
    print(f'Listed positions:{spaces}{list(zip(seasons, positions))}')

    # classify
    for classifier, name in trained_classifiers:
        predicted_positions = classifier.predict(player)

        spaces = ' ' * (max_name_len - len(name)) # for output formatting
        print(f'Predicted with {name}:{spaces} {list(zip(seasons, predicted_positions))}')
    
    print('') # new line


def classify_players(trained_classifiers, players=None, seasons_range=[2012, 2019]):
    '''
        Predict selected players from selected seasons. If players parameter is None,
        then players from the list below are selected.
        Season range should not include seasons from 1990-91 to 2010-11 because models
        are trained on those seasons!
    '''

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
            'Andre Drummond',
            'Ben Simmons',
            'Giannis Antetokounmpo',
        ]

    per_game = pd.read_csv(Path('../data/per_game_data.csv'))
    per_game = per_game[per_game.season >= seasons_range[0]]
    per_game = per_game[per_game.season <= seasons_range[1]]

    for player in players:
        classify_selected_player(per_game, player, trained_classifiers)


if __name__ == "__main__":
    df = pd.read_csv(Path('../data/per_game_data.csv'))

    df = filter_data(df)

    # visualize_data(df) # <- uncomment this for boxplots

    # split on data and label columns
    X = df.drop(columns=['Pos'])
    y = df['Pos']

    # split on train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)
    cv = StratifiedKFold(n_splits=5)

    # number of players by position
    print(f'Number of players by position in data is {dict(Counter(y))}\n')
    print(f'Number of players by position in train set is {dict(Counter(y_train))}\n')
    print(f'Number of players by position in test set is {dict(Counter(y_test))}\n')

    # try different parameters for different classifiers
    # find_best_classifiers(cv, X_train, X_test, y_train, y_test) #  <- uncomment this for grid search

    # create random classifier
    dummy = create_classifier(
        DummyClassifier(strategy='stratified'),
        'Dummy', X_train, X_test, y_train, y_test
    )

    # the best classifiers from GridSearchCV
    knn = create_classifier(
        KNeighborsClassifier(
            n_neighbors=10,
            weights='distance'
        ),
        'KNN', X_train, X_test, y_train, y_test
    )

    rfc = create_classifier(
        RandomForestClassifier(
            criterion='entropy',
            max_depth=20,
            n_estimators=300,
            random_state=27
        ),
        'Random Forest', X_train, X_test, y_train, y_test
    )

    gbc = create_classifier(
        GradientBoostingClassifier(
            loss='deviance',
            max_depth=20,
            max_features='sqrt',
            n_estimators=50,
            random_state=27
        ),
        'Gradient Boosting', X_train, X_test, y_train, y_test
    )

    svc = create_classifier(
        SVC(
            C=100,
            gamma=0.01,
            kernel='rbf'
        ),
        'SVC', X_train, X_test, y_train, y_test
    )

    # classify some of the players that play today (around 2018-19 season)
    classifiers = [
        (knn, 'KNN'), (rfc, 'Random Forest'),
        (svc, 'SVC'), (gbc, 'Gradient Boosting')
    ]
    classify_players(classifiers)
    
