"""Run Random Forest Regression on the linear model for various types of errors.

Store the data in checkpoint csvs, which can be merged into a single DataFrame
at a later time."""

import os
import glob
import json
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats, special
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
import sklearn.utils as sk_utils
from patsy import dmatrices

from utils import errors, models

# TODO: Refactor into functions (key concern is passing in arguments to noise
# functions).

#############################
# Run Gaussian noise model. #
#############################
results_cols = ['mse', 'oracle_mse', 'mae', 'oracle_mae']
results = []

# Find last checkpoint.
start = 1
search_start = start + 10
search_start_str = 'gaussian/rf_lin_test_gaussian_checkpoint_{}.csv'.format(
    search_start)
inc_start = False
while os.path.exists(search_start_str):
    inc_start = True
    start = search_start
    search_start += 10
    search_start_str = 'gaussian/rf_lin_test_gaussian_checkpoint_{}.csv'.format(
        search_start)
# Increment so we avoid overwriting a checkpoint.
if inc_start:
    start += 1

if not os.path.exists('gaussian'):
    os.mkdir('gaussian')
end = 501
if os.path.exists('gaussian/rf_lin_test_gaussian_checkpoint_last.csv'):
    start = end + 1
for i in range(start, end):
    df = models.simulate_10var_linear()
    # Add Gaussian noise.
    N = df.shape[0]
    df['noise'] = errors.generate_gaussian_noise(N, df['fX'].values)
    df['Y'] = df['fX'] + df['noise']

    train_df = df[df.index <= 399]
    test_df = df[df.index >= 400]

    # TODO: Should do some sort of hparam searching here. Can start with simple
    # random search.
    # Default options to test. See
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    rf_sample_options = {
        'max_features': 0.33,  # Setting in randomForest R package.
        'oob_score': True,
        'n_estimators': 1000,  # Start with 1000 trees.
    }

    rf = RandomForestRegressor(**rf_sample_options)
    features = ['X{}'.format(i) for i in range(1, 11)]
    rf.fit(train_df[features], train_df['Y'])

    pred = rf.predict(test_df[features])
    mse = np.mean(np.square(pred - test_df['Y']))
    oracle_mse = np.mean(np.square(test_df['fX'] - test_df['Y']))
    mae = np.mean(np.absolute(pred - test_df['Y']))
    oracle_mae = np.mean(np.absolute(test_df['fX'] - test_df['Y']))
    results.append([mse, oracle_mse, mae, oracle_mae])

    if i % 10 == 1 and i > 1:
        # Checkpoint every 10 iterations.
        results_df = pd.DataFrame(results, columns=results_cols)
        checkpoint = 'gaussian/rf_lin_test_gaussian_checkpoint_{}.csv'.format(i)
        results_df.to_csv(checkpoint, index=False)
        results = []
# Checkpoint at the end.
if len(results):
    results_df = pd.DataFrame(results, columns=results_cols)
    checkpoint = 'gaussian/rf_lin_test_gaussian_checkpoint_last.csv'
    results_df.to_csv(checkpoint, index=False)

###########################
# Run Cauchy noise model. #
###########################
results_cols = ['mse', 'oracle_mse', 'mae', 'oracle_mae']
results = []

# Find last checkpoint.
start = 1
search_start = start + 10
search_start_str = 'cauchy/rf_lin_test_cauchy_checkpoint_{}.csv'.format(
    search_start)
inc_start = False
while os.path.exists(search_start_str):
    inc_start = True
    start = search_start
    search_start += 10
    search_start_str = 'cauchy/rf_lin_test_cauchy_checkpoint_{}.csv'.format(
        search_start)
# Increment so we avoid overwriting a checkpoint.
if inc_start:
    start += 1

if not os.path.exists('cauchy'):
    os.mkdir('cauchy')
end = 501
if os.path.exists('cauchy/rf_lin_test_cauchy_checkpoint_last.csv'):
    start = end + 1
for i in range(start, end):
    df = models.simulate_10var_linear()
    # Add Cauchy noise.
    N = df.shape[0]
    df['noise'] = errors.generate_cauchy_noise(N, df['fX'].values)
    df['Y'] = df['fX'] + df['noise']

    train_df = df[df.index <= 399]
    test_df = df[df.index >= 400]

    # TODO: Should do some sort of hparam searching here. Can start with simple
    # random search.
    # Default options to test. See
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    rf_sample_options = {
        'max_features': 0.33,  # Setting in randomForest R package.
        'oob_score': True,
        'n_estimators': 1000,  # Start with 1000 trees.
    }

    rf = RandomForestRegressor(**rf_sample_options)
    features = ['X{}'.format(i) for i in range(1, 11)]
    rf.fit(train_df[features], train_df['Y'])

    pred = rf.predict(test_df[features])
    mse = np.mean(np.square(pred - test_df['Y']))
    oracle_mse = np.mean(np.square(test_df['fX'] - test_df['Y']))
    mae = np.mean(np.absolute(pred - test_df['Y']))
    oracle_mae = np.mean(np.absolute(test_df['fX'] - test_df['Y']))
    results.append([mse, oracle_mse, mae, oracle_mae])

    if i % 10 == 1 and i > 1:
        # Checkpoint every 10 iterations.
        results_df = pd.DataFrame(results, columns=results_cols)
        checkpoint = 'cauchy/rf_lin_test_cauchy_checkpoint_{}.csv'.format(i)
        results_df.to_csv(checkpoint, index=False)
        results = []
# Checkpoint at the end.
if len(results):
    results_df = pd.DataFrame(results, columns=results_cols)
    checkpoint = 'cauchy/rf_lin_test_cauchy_checkpoint_last.csv'
    results_df.to_csv(checkpoint, index=False)


##########################################
# Run Student's t noise model (DoF = 2). #
##########################################
results_cols = ['mse', 'oracle_mse', 'mae', 'oracle_mae']
results = []

# Find last checkpoint.
start = 1
search_start = start + 10
search_start_str = 'student_t/rf_lin_test_student_t_checkpoint_{}.csv'.format(
    search_start)
inc_start = False
while os.path.exists(search_start_str):
    inc_start = True
    start = search_start
    search_start += 10
    search_start_str = ('student_t/rf_lin_test_student_t_checkpoint_{}.csv'
        .format(search_start))
# Increment so we avoid overwriting a checkpoint.
if inc_start:
    start += 1

if not os.path.exists('student_t'):
    os.mkdir('student_t')
end = 501
if os.path.exists('student_t/rf_lin_test_student_t_checkpoint_last.csv'):
    start = end + 1
for i in range(start, end):
    df = models.simulate_10var_linear()
    # Add t-distributed noise.
    N = df.shape[0]
    df['noise'] = errors.generate_t2_noise(N, df['fX'].values)
    df['Y'] = df['fX'] + df['noise']

    train_df = df[df.index <= 399]
    test_df = df[df.index >= 400]

    # TODO: Should do some sort of hparam searching here. Can start with simple
    # random search.
    # Default options to test. See
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    rf_sample_options = {
        'max_features': 0.33,  # Setting in randomForest R package.
        'oob_score': True,
        'n_estimators': 1000,  # Start with 1000 trees.
    }

    rf = RandomForestRegressor(**rf_sample_options)
    features = ['X{}'.format(i) for i in range(1, 11)]
    rf.fit(train_df[features], train_df['Y'])

    pred = rf.predict(test_df[features])
    mse = np.mean(np.square(pred - test_df['Y']))
    oracle_mse = np.mean(np.square(test_df['fX'] - test_df['Y']))
    mae = np.mean(np.absolute(pred - test_df['Y']))
    oracle_mae = np.mean(np.absolute(test_df['fX'] - test_df['Y']))
    results.append([mse, oracle_mse, mae, oracle_mae])

    if i % 10 == 1 and i > 1:
        # Checkpoint every 10 iterations.
        results_df = pd.DataFrame(results, columns=results_cols)
        checkpoint = ('student_t/rf_lin_test_student_t_checkpoint_{}.csv'
            .format(i))
        results_df.to_csv(checkpoint, index=False)
        results = []
# Checkpoint at the end.
if len(results):
    results_df = pd.DataFrame(results, columns=results_cols)
    checkpoint = 'student_t/rf_lin_test_student_t_checkpoint_last.csv'
    results_df.to_csv(checkpoint, index=False)

##########################
# Run Skew Normal noise. #
##########################
results_cols = ['mse', 'oracle_mse', 'mae', 'oracle_mae']
results = []

# Find last checkpoint.
start = 1
search_start = start + 10
search_start_str = ('skew_normal/rf_lin_test_skew_normal_checkpoint_{}.csv'
    .format(search_start))
inc_start = False
while os.path.exists(search_start_str):
    inc_start = True
    start = search_start
    search_start += 10
    search_start_str = ('skew_normal/rf_lin_test_skew_normal_checkpoint_{}.csv'
        .format(search_start))
# Increment so we avoid overwriting a checkpoint.
if inc_start:
    start += 1

if not os.path.exists('skew_normal'):
    os.mkdir('skew_normal')
end = 501
if os.path.exists('skew_normal/rf_lin_test_skew_normal_checkpoint_last.csv'):
    start = end + 1
for i in range(start, end):
    df = models.simulate_10var_linear()
    # Add skew normal noise.
    N = df.shape[0]
    df['noise'] = errors.generate_skew_normal_errors(-10, N, df['fX'].values)
    df['Y'] = df['fX'] + df['noise']

    train_df = df[df.index <= 399]
    test_df = df[df.index >= 400]

    # TODO: Should do some sort of hparam searching here. Can start with simple
    # random search.
    # Default options to test. See
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    rf_sample_options = {
        'max_features': 0.33,  # Setting in randomForest R package.
        'oob_score': True,
        'n_estimators': 1000,  # Start with 1000 trees.
    }

    rf = RandomForestRegressor(**rf_sample_options)
    features = ['X{}'.format(i) for i in range(1, 11)]
    rf.fit(train_df[features], train_df['Y'])

    pred = rf.predict(test_df[features])
    mse = np.mean(np.square(pred - test_df['Y']))
    oracle_mse = np.mean(np.square(test_df['fX'] - test_df['Y']))
    mae = np.mean(np.absolute(pred - test_df['Y']))
    oracle_mae = np.mean(np.absolute(test_df['fX'] - test_df['Y']))
    results.append([mse, oracle_mse, mae, oracle_mae])

    if i % 10 == 1 and i > 1:
        # Checkpoint every 10 iterations.
        results_df = pd.DataFrame(results, columns=results_cols)
        checkpoint = ('skew_normal/rf_lin_test_skew_normal_checkpoint_{}.csv'
            .format(i))
        results_df.to_csv(checkpoint, index=False)
        results = []
# Checkpoint at the end.
if len(results):
    results_df = pd.DataFrame(results, columns=results_cols)
    checkpoint = 'skew_normal/rf_lin_test_skew_normal_checkpoint_last.csv'
    results_df.to_csv(checkpoint, index=False)

##################
# Run GEV noise. #
##################
results_cols = ['mse', 'oracle_mse', 'mae', 'oracle_mae']
results = []

# Find last checkpoint.
start = 1
search_start = start + 10
search_start_str = ('gev/rf_lin_test_gev_checkpoint_{}.csv'
    .format(search_start))
inc_start = False
while os.path.exists(search_start_str):
    inc_start = True
    start = search_start
    search_start += 10
    search_start_str = ('gev/rf_lin_test_gev_checkpoint_{}.csv'
        .format(search_start))
# Increment so we avoid overwriting a checkpoint.
if inc_start:
    start += 1

if not os.path.exists('gev'):
    os.mkdir('gev')
end = 501
if os.path.exists('gev/rf_lin_test_gev_checkpoint_last.csv'):
    start = end + 1
for i in range(start, end):
    df = models.simulate_10var_linear()
    # Add gev noise.
    N = df.shape[0]
    df['noise'] = errors.generate_gev_noise(2, N, df['fX'].values)
    df['Y'] = df['fX'] + df['noise']

    train_df = df[df.index <= 399]
    test_df = df[df.index >= 400]

    # TODO: Should do some sort of hparam searching here. Can start with simple
    # random search.
    # Default options to test. See
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    rf_sample_options = {
        'max_features': 0.33,  # Setting in randomForest R package.
        'oob_score': True,
        'n_estimators': 1000,  # Start with 1000 trees.
    }

    rf = RandomForestRegressor(**rf_sample_options)
    features = ['X{}'.format(i) for i in range(1, 11)]
    rf.fit(train_df[features], train_df['Y'])

    pred = rf.predict(test_df[features])
    mse = np.mean(np.square(pred - test_df['Y']))
    oracle_mse = np.mean(np.square(test_df['fX'] - test_df['Y']))
    mae = np.mean(np.absolute(pred - test_df['Y']))
    oracle_mae = np.mean(np.absolute(test_df['fX'] - test_df['Y']))
    results.append([mse, oracle_mse, mae, oracle_mae])

    if i % 10 == 1 and i > 1:
        # Checkpoint every 10 iterations.
        results_df = pd.DataFrame(results, columns=results_cols)
        checkpoint = ('gev/rf_lin_test_gev_checkpoint_{}.csv'
            .format(i))
        results_df.to_csv(checkpoint, index=False)
        results = []
# Checkpoint at the end.
if len(results):
    results_df = pd.DataFrame(results, columns=results_cols)
    checkpoint = 'gev/rf_lin_test_gev_checkpoint_last.csv'
    results_df.to_csv(checkpoint, index=False)