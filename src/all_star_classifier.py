#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import shap

import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


def plot_one_distplot(data, target, xlabel, title):
    sns.distplot(df[df['ALL_STAR'] == 1][target])
    sns.distplot(df[df['ALL_STAR'] == 0][target])
    plt.legend(labels=['All-Stars', 'Not All-Stars'], fontsize=20)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(xlabel, fontsize=20)

    plt.title(title, fontsize=28)
    plt.show()


def plot_distplots(data):
    plot_one_distplot(
        data, 'PTS',
        'Points scored per game',
        'Distribution of PTS per game'
    )

    plot_one_distplot(
        data, 'TRB',
        'Number of Rebounds per game',
        'Distribution of TRB per game'
    )

    plot_one_distplot(
        data, 'AST',
        'Number of assists per game',
        'Distribution of AST per game'
    )

    plot_one_distplot(
        data, 'DraftPick',
        'Draft position',
        'Distribution of draft positions'
    )


def plot_boxplots(data):
    label_size = 18
    ticks_size = 18
    subtitle_size = 20

    fig, axs = plt.subplots(nrows=2, ncols=2)

    sns.boxplot(data['ALL_STAR'], data['DraftPick'], ax=axs[0][0])
    axs[0][0].title.set_text('Draft Pick')
    axs[0][0].title.set_fontsize(subtitle_size)
    axs[0][0].tick_params(labelsize=ticks_size)
    axs[0][0].set_xlabel('')
    axs[0][0].set_ylabel('Pick', fontsize=label_size)

    sns.boxplot(data['ALL_STAR'], data['PTS'], ax=axs[0][1])
    axs[0][1].title.set_text('Points per game')
    axs[0][1].title.set_fontsize(subtitle_size)
    axs[0][1].tick_params(labelsize=ticks_size)
    axs[0][1].set_xlabel('')
    axs[0][1].set_ylabel('PPG', fontsize=label_size)

    sns.boxplot(data['ALL_STAR'], data['OBPM'], ax=axs[1][0])
    axs[1][0].title.set_text('Offensive box plus/minus')
    axs[1][0].title.set_fontsize(subtitle_size)
    axs[1][0].tick_params(labelsize=ticks_size)
    axs[1][0].set_xlabel('')
    axs[1][0].set_ylabel('OBPM', fontsize=label_size)

    sns.boxplot(data['ALL_STAR'], data['DBPM'], ax=axs[1][1])
    axs[1][1].title.set_text('Defensive box plus/minus')
    axs[1][1].title.set_fontsize(subtitle_size)
    axs[1][1].tick_params(labelsize=ticks_size)
    axs[1][1].set_xlabel('')
    axs[1][1].set_ylabel('DBPM', fontsize=label_size)

    fig.suptitle('Distribution of All-Stars (1) and not All-Stars (0)', fontsize=30)
    plt.show()


def filter_data(df):
    # transform data (totals in a season)
    # into per game data
    df['PTS'] = np.around(df['PTS'] / df['G'], 3)
    df['TRB'] = np.around(df['TRB'] / df['G'], 3)
    df['AST'] = np.around(df['AST'] / df['G'], 3)
    df['STL'] = np.around(df['STL'] / df['G'], 3)
    df['BLK'] = np.around(df['BLK'] / df['G'], 3)
    df['MP'] = np.around(df['MP'] / df['G'], 3)

    filtered_df = df[[
        'TS%', '3PAr', 'FTr',
        'OBPM', 'DBPM', 'TRB',
        'AST', 'STL', 'BLK',
        'PTS', 'DraftPick',
        'ALL_STAR'
    ]]
    
    data = filtered_df.drop(columns=['ALL_STAR'])
    target = filtered_df['ALL_STAR']

    # data = MinMaxScaler(feature_range=(0, 100)).fit_transform(data) # <- uncoment this and import for scaling
    return data, target, filtered_df


def best_classifier(model, grid, model_name, cv, X_train, X_test, y_train, y_test):
    clf = GridSearchCV(model, param_grid=grid, scoring='accuracy', cv=cv, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f'Best params for {model_name}: {clf.best_params_}')
    print(f'Best score for {model_name}: {clf.best_score_}')
    print('') # new line


def find_best_classifiers(cv, X_train, X_test, y_train, y_test):
    '''
        Run different classification algorithms
        with different parameters. Grid search is used.
        Best models are then created separately. 
    '''

    best_classifier(
        SVC(),
        {
            'kernel': ['poly'],
            'degree': [i for i in range(1, 11)],
            'gamma': ['scale'],
            'C': [0.1, 1, 10, 100],
            'probability': [True]
        },
        'Poly  SVC', cv,
        X_train, X_test, y_train, y_test
    )

    best_classifier(
        SVC(),
        {
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale'],
            'C': [0.1, 1, 10, 100],
            'probability': [True]
        },
        'Rbf and Linear SVC', cv,
        X_train, X_test, y_train, y_test
    )

    best_classifier(
        RandomForestClassifier(),
        {
            'n_estimators': np.linspace(start=50, stop=300, num=6, dtype=int),
            'criterion': ['gini'],
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


def create_classifier(model, model_name, X_train, X_test, y_train, y_test, scoring):
    clf = model
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'{model_name} train accuracy: {clf.score(X_train, y_train):.2f}')
    print(f'{model_name} test accuracy: {clf.score(X_test, y_test):.2f}')

    proba = clf.predict_proba(X_test)
    print(f'Log loss: {log_loss(y_test, proba):.3f}')

    pos_prob = proba[:, 1]
    print(f'Area under ROC curve: {roc_auc_score(y_test, pos_prob):.2f}')

    score = cross_val_score(clf, X_test, y_test, cv=5, scoring=scoring)
    print(f'{model_name} cross-validation score: {score.mean():.2f} (+/- {(score.std() * 2):.2f})')

    print(f'{model_name} classification report: \n{classification_report(y_test, y_pred)}')
    print(f'{model_name} confusion matrix: \n{confusion_matrix(y_test, y_pred)}')

    print('\n------------------------------\n') # separator

    return clf


def apply_models_to_data(models):
    df = pd.read_csv(Path('../data/rookies_all_stars_test_data.csv'))
    names = df['Player'].values

    X, y, filtered_df = filter_data(df)

    results = []
    for model, name in models:
        print(name + ' results:')
        result = model.predict_proba(X)

        # [:, 1] is an All-Star probability, [:, 0] is for not All-Star probability
        print(np.around(result[:, 1], 3))
        print('-----------------')

        results.append(result[:, 1])

    # calculate average probability
    print('average probability:')
    results = np.array(results)
    probabilites = results.mean(axis=0)

    for name, probability in zip(names, probabilites):
        print(name, np.around(probability, 2), sep=': ')

    print('-------------------------')


def SHAP_values(model, X_train):
    k_sample = shap.kmeans(X_train, 5)
    explainer = shap.KernelExplainer(model.predict, k_sample)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, plot_type='bar')
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(Path('../data/rookies_data_2001_2017.csv'))

    X, y, filtered_df = filter_data(df)

    # various distribution plots
    # plot_boxplots(filtered_df)
    # plot_distplots(filtered_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

    print(f'Number of All-stars is: {Counter(y)}')
    print(f'Number of All-stars in train set is: {Counter(y_train)}')
    print(f'Number of All-stars in test set is: {Counter(y_test)}')

    cv = StratifiedKFold(n_splits=5)
    # find_best_classifiers(cv, X_train, X_test, y_train, y_test) # <- uncomment for Grid Search

    # dummy (random) classifier
    create_classifier(
        DummyClassifier(
            strategy='stratified',
            random_state=27
        ), 'Dummy',
        X_train, X_test, y_train, y_test, 'accuracy'
    )

    # maximizing accuracy:
    svc_acc = create_classifier(
        SVC(
            kernel='poly',
            gamma='scale',
            C=1,
            degree=1,
            probability=True
        ), 'SVC',
        X_train, X_test, y_train, y_test, 'accuracy'
    )

    # SHAP_values(svc_acc, X_train)

    rfc_acc = create_classifier(
        RandomForestClassifier(
            criterion='gini',
            max_depth=10,
            n_estimators=100,
            random_state=27
        ), 'RFC',
        X_train, X_test, y_train, y_test, 'accuracy'
    )

    # SHAP_values(rfc_acc, X_train)

    gbc_acc = create_classifier(
        GradientBoostingClassifier(
            loss='deviance',
            max_depth=10,
            max_features='sqrt',
            n_estimators=50,
            random_state=27
        ), 'GBC',
        X_train, X_test, y_train, y_test, 'accuracy'
    )

    # SHAP_values(gbc_acc, X_train)

    # maximizing recall:
    svc_rcl = create_classifier(
        SVC(
            kernel='poly',
            gamma='scale',
            C=100,
            degree=1,
            probability=True
        ), 'SVC',
        X_train, X_test, y_train, y_test, 'recall'
    )

    # SHAP_values(svc_rcl, X_train)

    rfc_rcl = create_classifier(
        RandomForestClassifier(
            criterion='gini',
            max_depth=10,
            n_estimators=200,
            random_state=27
        ), 'RFC',
        X_train, X_test, y_train, y_test, 'recall'
    )

    # SHAP_values(rfc_rcl, X_train)

    gbc_rcl = create_classifier(
        GradientBoostingClassifier(
            loss='deviance',
            max_depth=10,
            max_features=None,
            n_estimators=100,
            random_state=27
        ), 'GBC',
        X_train, X_test, y_train, y_test, 'recall'
    )

    # SHAP_values(gbc_rcl, X_train)

    print('Accuracy:')
    apply_models_to_data([(svc_acc, 'SVC'), (rfc_acc, 'RFC'), (gbc_acc, 'GBC')])

    print('Recall:')
    apply_models_to_data([(svc_rcl, 'SVC'), (rfc_rcl, 'RFC'), (gbc_rcl, 'GBC')])

    print('Both:')
    apply_models_to_data([
        (svc_acc, 'SVC acc'), (rfc_acc, 'RFC acc'), (gbc_acc, 'GBC acc'),
        (svc_rcl, 'SVC rcl'), (rfc_rcl, 'RFC rcl'), (gbc_rcl, 'GBC rcl')
    ])
