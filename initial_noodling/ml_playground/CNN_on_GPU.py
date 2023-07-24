# RUNS ON GPU #0
#
# uses 4 workers

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values


## DATA

all_data = pd.read_csv("https://data.ecoforecast.org/neon4cast-targets/beetles/beetles-targets.csv.gz")
all_data["datetime"] = pd.to_datetime(all_data["datetime"]).dt.tz_localize(None)
stei_data_raw = all_data.loc[all_data["site_id"] == "STEI"]
tree_data_raw = all_data.loc[all_data["site_id"] == "TREE"]
unde_data_raw = all_data.loc[all_data["site_id"] == "UNDE"]

stei_data = stei_data_raw[["datetime", "site_id", "variable", "observation"]].pivot(index="datetime", columns="variable", values="observation")
tree_data = tree_data_raw[["datetime", "site_id", "variable", "observation"]].pivot(index="datetime", columns="variable", values="observation")
unde_data = unde_data_raw[["datetime", "site_id", "variable", "observation"]].pivot(index="datetime", columns="variable", values="observation")

stei_data.columns = ['abundance', 'richness']
tree_data.columns = ['abundance', 'richness']
unde_data.columns = ['abundance', 'richness']

stei_series = TimeSeries.from_dataframe(
    stei_data,
    freq = "D",
)

tree_series = TimeSeries.from_dataframe(
    tree_data,
    freq = "D",
)

unde_series = TimeSeries.from_dataframe(
    unde_data,
    freq = "D",
)

stei_series_filled = fill_missing_values(stei_series)
tree_series_filled = fill_missing_values(tree_series)
unde_series_filled = fill_missing_values(unde_series)



### DATA SERIES THAT WILL ACTUALLY BE CALLED

_series = tree_series_filled[["abundance"]]

_train_frac = 0.75
_train, _val = _series.split_before(_train_frac)


### HYPERPARAMETER TUNING

from pytorch_lightning.callbacks import Callback
import torch
from darts.models import TCNModel

def fit_model(
    in_len,
    out_len,
    kernel_size,
    num_filters,
    weight_norm,
    dilation_base,
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
    MAX_N_EPOCHS = 40
    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000
    
    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping("train_loss", min_delta=0.0001, patience=3, verbose=True)
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
    
    # if torch.cuda.is_available():
    #     pl_trainer_kwargs = {
    #         "accelerator": "gpu",
    #         "callbacks": callbacks,
    #     }
    #     num_workers = 4
    
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_N_EPOCHS,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        likelihood=likelihood,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
    )
    
    # fit model to train data
    model.fit(
        series = _train,
        max_samples_per_ts = MAX_SAMPLES_PER_TS,
        num_loader_workers = num_workers,
    )
    
    return model
    
from pytorch_lightning.callbacks import Callback, EarlyStopping
from darts.metrics import marre
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial):
    # trial is not just a darts model, it's an Optuna object which I guess
    # keeps track of the parameters that it will try now (with the "suggest"
    # methods).
    
    callback = [PyTorchLightningPruningCallback(trial, monitor="train_loss")]
    
    # set input_chunk_length, between 5 and 14 days
    input_chunk_length = trial.suggest_int("input_chunk_length", 300, 400)

    # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
    output_chunk_length = trial.suggest_int("output_chunk_length", 150, input_chunk_length - 1)

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 5, 25)
    num_filters = trial.suggest_int("num_filters", 5, 25)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)

    # build and train the TCN model with these hyper-parameters:
    model = fit_model(
        in_len=input_chunk_length,
        out_len=output_chunk_length,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        lr=lr,
        callbacks=callback,
    )

    # Evaluate how good it is on the validation set
    preds = model.predict(series = _train, n = len(_val) )
    metric_values = marre(_val, preds, n_jobs=-1, verbose=True)
    mean_metric_value = np.mean(metric_values)

    return mean_metric_value if mean_metric_value != np.nan else float("inf")

def print_callback(study, trial):
    # For keeping track while optimizing
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    

import optuna

study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=7200, callbacks=[print_callback])   