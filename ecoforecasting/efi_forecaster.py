from datetime import datetime

import os

os.makedirs(DATAPATH, exist_ok=True)

from darts.models.forecasting.forecasting_model import ForecastingModel 
# the base forecasting model class

from ecoforecasting import named_model

NR_PREDS_A_DAY = { #TBD
	"terrestrial": 1,
	"aquatics": 1,
	"tick": 1,
	"phenology": 1,
	"beetle": 1,
}

HORIZON_DAYS = {
	"day": 1,
	"week": 7,
	"month": 31,
}

class efi_forecaster:
	""" uses trained/fitted model to produce forecast in EFI NEON format """

	def __init__(
		self, 
		model: named_model, 
		theme: str = "terrestrial",
		forecast_horizon: str = "month",
	):
		"""
		model: a named model (wrapper of ForecastingModel)
		challenge: 'terrestrial', 'aquatics', 'tick', 'phenology', 'beetle'
		forecast_horizon: 'day', 'week', 'month' [31 days]
		"""
		self.model = model.model
		self.model_name = model.model_name
		self.theme = theme
		self.forecast_horizon = forecast_horizon

	def csv_forecast(
		self, 
		start_datetime: datetime,
		target_dir: str = os.path.join("..", "data", f"{self.model_name}")
	):
		"""
		produces csv file with the format needed for EFI submissions.
		"""

		os.makedirs(target_dir, exist_ok=True)
		fname = f"{self.challenge}-{start_datetime}-{self.model_name}"
		path_and_fname = os.path.join(target_dir, fname)

		horizon_n = self._get_horizon_n

	def _get_horizon_n(self):
		""" how many individual predictions in a forecast horizon """
		return NR_PREDS_A_DAY[self.theme] * HORIZON_DAYS[self.forecast_horizon]


