import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.metrics import marre, mse
from darts.models.forecasting.forecasting_model import ForecastingModel # the base forecasting model class

from dataclasses import dataclass
from typing import callable

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch

from DataProcessing import NOAA_stage3_scan, day_mean_several, quick_neon_series, get_noaa

@dataclass
class named_model:
	model: ForecastingModel
	model_name: str

class hyperparam_tuner:
	""" tunes hyperparameters of a model """
	def __init__(
		model: named_model,
		suggest_params_fn: Callable,
		callbacks = None,
		use_past_cov: bool = False,
		use_futu_cov: bool = False,
		max_samples_per_ts: int = 1000,
		num_workers: int = 4,
	):
		self.model = model # model whose hyperparams are tuned
		self.model_name = model_name
		self.callbacks = callbacks
		self.static_hyperparams = static_hyperparams
		self.max_samples_per_ts = max_samples_per_ts
		self.num_workers = num_workers

		self.pl_trainer_kwargs = {
			"accelerator": "gpu",
			"devices": [0],
			"callbacks": self.callbacks,
		}
	
	def fit_model(
		self,
		series,
		variable_hyperparams: dict, 
		**kwargs,
	):
		"""
		fits self.model to a series, optionally with covariates given as
		kwargs:
			model_kwargs:
				add_encoders: dict (see Darts models docs for format)
			fit_kwargs:
				future_covariates: series
				past_covariates: series
		"""

		# reproducibility
		torch.manual_seed(42)

		model_kwargs = kwargs.get("model_kwargs", {})
		fit_kwargs = kwargs.get("fit_kwargs", {})

		self.model(
			model_name=self.model_name,
			force_reset=True,
			save_checkpoints=True,
			pl_trainer_kwargs=self.pl_trainer_kwargs,
			**self.static_hyperparams,
			**variable_hyperparams,
			**model_kwargs,
		).fit(
			series = series,
			max_samples_per_ts = self.max_samples_per_ts,
			num_loader_workers = self.num_workers,
			**fit_kwargs,
		)

		return self.model


	def objective(self, trial, series, val_series, **fit_kwargs, **model_kwargs):
		""" trial is an optuna object """
		callback = [PyTorchLightningPruningCallback(trial, monitor="train_loss")]

		variable_params = self.suggest_params_fn()

		self.model = self.fit_model(
			series=series,
			variable_hyperparams=variable_hyperparams, 
			**fit_kwargs, 
			**model_kwargs,
		)

		preds = self.model.predict(series = series, n = len(val_series) )
		metric_values = mse(val_series, preds, n_jobs=-1, verbose=True)
		mean_metric_value = np.mean(metric_values)

		return mean_metric_value if mean_metric_value != np.nan else float("inf")


