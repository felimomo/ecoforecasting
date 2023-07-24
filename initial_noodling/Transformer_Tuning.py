# RUNS ON GPU #1
#
# uses 4 workers

# fixed input chunk, made output chunk have smaller ranges around a year.

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.metrics import marre, mse
from darts.models import TransformerModel

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

# train val
date_cutoff = pd.Timestamp("2023-02-26")
past_noaa, future_noaa = noaa_covariates.split_before(date_cutoff)
train_konz, val_konz = konz.split_before(date_cutoff)
train_kona, val_kona = kona.split_before(date_cutoff)
train_ukfs, val_ukfs = ukfs.split_before(date_cutoff)

## DEPENDS ON CHOICE OF SITE KONZ
covariates_konz = (
    past_noaa
    .concatenate(train_kona, axis = "component")
    .concatenate(train_ukfs[['le_day_avg']], axis = "component")
)


### HYPERPARAMETER TUNING

def fit_model(
    in_len,
    out_len,
    d_model,
    nhead,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
    lr,
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
    
    add_encoders={
        'cyclic': {'future': ['month']},
    }
    
    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping("train_loss", min_delta=0.0001, patience=5, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks
    
    # detect if a GPU is available
    pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": [0],
            "callbacks": callbacks,
        }
    num_workers = 4
    
    model = TransformerModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        #
        add_encoders = add_encoders,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_N_EPOCHS,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        #
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout = dropout,
        optimizer_kwargs={"lr": lr},
        #
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="transformer_model",
        force_reset=True,
        save_checkpoints=True,
    )
    
    # fit model to train data
    model.fit(
        series = train_konz,
        past_covariates = covariates_konz,
        max_samples_per_ts = MAX_SAMPLES_PER_TS,
        num_loader_workers = num_workers,
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
    output_chunk_length = trial.suggest_int("output_chunk_length", 15, 35)

    # Other hyperparameters
    d_model = trial.suggest_int("d_model", 56, 72)
    nhead = trial.suggest_categorical("nhead", [2, 4])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 4)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 4)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.2, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)

    # build and train the TFT model with these hyper-parameters:
    model = fit_model(
        in_len=input_chunk_length,
        out_len=output_chunk_length,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        lr = lr,
        callbacks=callback,
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