from ecoforecasting.tuning import hyperparam_tuner
# from ecoforecasting.named_model import named_model_class
from ecoforecasting.params_suggestions import TranformerSugg
from ecoforecasting.data_processing import NOAA_stage3_scan, day_mean_several, quick_neon_series, get_noaa

from pytorch_lightning.callbacks import Callback, EarlyStopping

from darts.models import TransformerModel

import optuna
import pandas as pd

# data globals

AQUATICS = "https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-expanded-observations.csv.gz"
TERRESTRIAL = "https://data.ecoforecast.org/neon4cast-targets/terrestrial_30min/terrestrial_30min-targets.csv.gz"
TICK = "https://data.ecoforecast.org/neon4cast-targets/ticks/ticks-targets.csv.gz"
PHENOLOGY = "https://data.ecoforecast.org/neon4cast-targets/phenology/phenology-targets.csv.gz"
BEETLE = "https://data.ecoforecast.org/neon4cast-targets/beetles/beetles-targets.csv.gz"

konz = quick_neon_series(
    site_id = "KONZ",
    link = TERRESTRIAL,
    freq = "D",
    time_col = "datetime",
    day_avg= True,
)


noaa_covariates = get_noaa(site_id = "KONZ")

date_cutoff = pd.Timestamp("2023-06-01")
PAST_COVARIATES, FUTURE_COVARIATES = noaa_covariates.split_before(date_cutoff)
SERIES, VAL_SERIES = konz.split_before(date_cutoff)

print(f"training series length: {len(SERIES)}")
print(f"val series length: {len(VAL_SERIES)}")

# fixed hyperparams, other globals

STATIC_PARAMS = {
	"batch_size": 1024,
	"n_epochs": 20,
	"nr_epochs_val_period": 1,
	"input_chunk_length": 356,
}

ADD_ENCODERS={
	'cyclic': {'future': ['month']},
}

MODEL_KWARGS = {"add_encoders": ADD_ENCODERS}
FIT_KWARGS = {
	"past_covariates": PAST_COVARIATES, 
	"max_samples_per_ts": 1000, 
	"num_loader_workers": 4,
}
KWARGS = {"model_kwargs": MODEL_KWARGS, "fit_kwargs": FIT_KWARGS}

# initializing objects

# transformer = named_model_class(model=TransformerModel, model_name="TransformerModel")
# print(transformer)
transformer_tuner = hyperparam_tuner(
	model=TransformerModel,
	model_name="TransformerModel",
	suggest_params_fn=TranformerSugg,
	static_hyperparams=STATIC_PARAMS,
)

tuned_transformer_instance = transformer_tuner.tuned_model_instance(
	series = SERIES, 
	val_series = VAL_SERIES,
	**KWARGS,
)

# obj_wrapper = lambda trial: transformer_tuner.objective(trial, SERIES, VAL_SERIES, model_kwargs=MODEL_KWARGS, fit_kwargs=FIT_KWARGS)

# study = optuna.create_study(direction="minimize")
# study.optimize(obj_wrapper, timeout=7200, callbacks=None)  
