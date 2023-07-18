# RUNS ON GPU #1
#
# uses 4 workers

# fixed input chunk, made output chunk have smaller ranges around a year.

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.metrics import marre, mse
from darts.models import RandomForest

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch

from DataProcessing import NOAA_stage3_scan, day_mean_several, quick_neon_series, get_noaa

## GLOBALS

AQUATICS = "https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-expanded-observations.csv.gz"
TERRESTRIAL = "https://data.ecoforecast.org/neon4cast-targets/terrestrial_30min/terrestrial_30min-targets.csv.gz"
TICK = "https://data.ecoforecast.org/neon4cast-targets/ticks/ticks-targets.csv.gz"
PHENOLOGY = "https://data.ecoforecast.org/neon4cast-targets/phenology/phenology-targets.csv.gz"
BEETLE = "https://data.ecoforecast.org/neon4cast-targets/beetles/beetles-targets.csv.gz"


## GLOBAL DATA

konz = quick_neon_series(
    site_id = "KONZ",
    link = TERRESTRIAL,
    freq = "D",
    time_col = "datetime",
    day_avg= True,
)

kona = quick_neon_series(
    site_id = "KONA",
    link = TERRESTRIAL,
    freq = "D",
    time_col = "datetime",
    day_avg= True,
)

# just use 'le' subseries as covariate, 'nee' is too noisy here
ukfs = quick_neon_series(
    site_id = "UKFS",
    link = TERRESTRIAL,
    freq = "D",
    time_col = "datetime",
    day_avg= True,
)

noaa_covariates = get_noaa(site_id = "KONZ")

# train, val
#
date_cutoff = pd.Timestamp("2023-02-20")
train_konz, val_konz = konz.split_before(date_cutoff)
#
past_noaa, future_noaa = noaa_covariates.split_after(date_cutoff)
train_kona, val_kona = kona.split_after(date_cutoff)
train_ukfs, val_ukfs = ukfs.split_after(date_cutoff)

## DEPENDS ON CHOICE OF SITE KONZ
covariates_konz = (
    past_noaa
    .concatenate(train_kona, axis = "component")
    .concatenate(train_ukfs[['le_day_avg']], axis = "component")
)


### HYPERPARAMETER TUNING

def fit_model(
    lags,
    lags_past_covariates,
    output_chunk_length,
    n_estimators,
    max_depth,
    likelihood=None,
    callbacks=None,
):
    # train parameter is the training series
    
    # reproducibility
    torch.manual_seed(42)

    # some fixed parameters that will be the same for all models
    BATCH_SIZE = 1024
    MAX_N_EPOCHS = 70
    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000
    
    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping("train_loss", min_delta=0.0001, patience=5, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks
    
    model = RandomForest(
        output_chunk_length = output_chunk_length,
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    
    # fit model to train data
    model.fit(
        series = train_konz,
        past_covariates = covariates_konz,
        max_samples_per_ts = MAX_SAMPLES_PER_TS,
    )
    
    return model


def objective(trial):
    # trial is not just a darts model, it's an Optuna object which I guess
    # keeps track of the parameters that it will try now (with the "suggest"
    # methods).
    
    callback = [PyTorchLightningPruningCallback(trial, monitor="train_loss")]
    
    # set input length to a year
    # input_chunk_length = trial.suggest_int("input_chunk_length", 330, 380)
    input_chunk_length = 356
    
    # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
    output_chunk_length = trial.suggest_int("output_chunk_length", 30, 40)

    # Other hyperparameters    
    lags = trial.suggest_int("lags", 50, 100)
    lags_past_covariates = trial.suggest_int("lags_past_covariates", 50, 100)
    n_estimators = trial.suggest_int("n_estimators", 91, 121)
    max_depth = trial.suggest_int("max_depth", 3, 10)


    # build and train the TFT model with these hyper-parameters:
    model = fit_model(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        output_chunk_length=output_chunk_length,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Evaluate how good it is on the validation set
    preds = model.predict(series = train_konz, n = len(val_konz) )
    metric_values = mse(val_konz, preds, n_jobs=-1, verbose=True)
    mean_metric_value = np.mean(metric_values)

    return mean_metric_value if mean_metric_value != np.nan else float("inf")

def print_callback(study, trial):
    # For keeping track while optimizing
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    

study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=7200, callbacks=[print_callback])   