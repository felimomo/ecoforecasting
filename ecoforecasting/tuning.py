import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.metrics import marre, mse
from darts.models.forecasting.forecasting_model import ForecastingModel 

from typing import Callable, Type

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch

from ecoforecasting.data_processing import ( 
	NOAA_stage3_scan, day_mean_several, quick_neon_series, get_noaa
)

# from ecoforecasting import named_model_class

class hyperparam_tuner:
	""" tunes hyperparameters of a model """

	def __init__(
		self,
		model: Type[ForecastingModel],
		model_name: str,
		suggest_params_fn: Callable,
		static_hyperparams: dict,
	):
		self.model = model # model whose hyperparams are tuned
		self.model_name = model_name
		self.suggest_params_fn = suggest_params_fn
		self.static_hyperparams = static_hyperparams

		self.pl_trainer_kwargs = {
			"accelerator": "gpu",
			"devices": [0],
			"callbacks": None,
		}
	
	def model_instance(self, variable_hyperparams: dict, **model_kwargs):
		""" 
		returns a model instance for given hyperparams and model_kwargs.
		"""
		return self.model(
			model_name=self.model_name,
			force_reset=True,
			save_checkpoints=True,
			pl_trainer_kwargs=self.pl_trainer_kwargs,
			**self.static_hyperparams,
			**variable_hyperparams,
			**model_kwargs,
		)

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

		fitted_model_instance = self.model_instance(
			variable_hyperparams,
			**model_kwargs,
		).fit(
			series = series,
			**fit_kwargs,
		)

		return fitted_model_instance


	def objective(self, trial, series, val_series, **kwargs):
		""" trial is an optuna object """

		# callbacks
		early_stopper = EarlyStopping("train_loss", min_delta=0.0001, patience=5, verbose=True)
		callback = [early_stopper] + [PyTorchLightningPruningCallback(trial, monitor="train_loss")]
		self.pl_trainer_kwargs["callbacks"] = callback

		# trial hyperparm suggestion
		variable_hyperparams = self.suggest_params_fn(trial)

		# fit
		fitted_model_instance = self.fit_model(
			series=series,
			variable_hyperparams=variable_hyperparams, 
			**kwargs,
		)

		out_chunk_ln = variable_hyperparams["output_chunk_length"]


		# eval
		preds = fitted_model_instance.predict(series = series, n = min(len(val_series), out_chunk_ln) )
		metric_values = mse(val_series, preds, n_jobs=-1, verbose=True)
		mean_metric_value = np.mean(metric_values)

		return mean_metric_value if mean_metric_value != np.nan else float("inf")

	def find_best_params(self, series, val_series, **kwargs):
		""" perform hyperparameter tuning """

		obj_wrapper = lambda trial: self.objective(
			trial, 
			series, 
			val_series, 
			model_kwargs = kwargs.get("model_kwargs", {}),
			fit_kwargs = kwargs.get("fit_kwargs", {}),
		)
		study = optuna.create_study(direction="minimize")
		study.optimize(obj_wrapper, timeout=7200, callbacks=None)
		return study.best_params

	def tuned_model_instance(self, series, val_series, **kwargs):
		""" return tuned model """

		opt_variable_hyperparams = self.find_best_params(series, val_series, **kwargs)
		model_kwargs = kwargs.get("model_kwargs", {})

		return self.model(
			model_name=self.model_name,
			force_reset=True,
			save_checkpoints=True,
			pl_trainer_kwargs=self.pl_trainer_kwargs,
			**self.static_hyperparams,
			**opt_variable_hyperparams,
			**model_kwargs,
		)



