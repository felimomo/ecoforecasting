import os
from datetime import datetime
from typing import Type

from darts.models.forecasting.forecasting_model import ForecastingModel 
# the base forecasting model class, used for type hints only
from darts import TimeSeries

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
		theme: str = "terrestrial",
		model: Type[ForecastingModel],
		model_name: str = "ForecastingModel",
		forecast_horizon: str = "month",
	):
		self.theme = theme
		self.model = model
		self.model_name = model_name
		self.forecast_horizon = forecast_horizon

	def csv_forecast(
		self, 
		series: TimeSeries,
		start_datetime: datetime,
		target_dir: str = os.path.join("..", "data", f"{self.model_name}")
	):
		"""
		produces csv file with the format needed for EFI submissions.
		"""

		# data path
		os.makedirs(target_dir, exist_ok=True)
		fname = f"{self.challenge}-{start_datetime}-{self.model_name}"
		path_and_fname = os.path.join(target_dir, fname)

		# produce forecast
		horizon_n = self._get_horizon_n()
		forecast = self.model.predict(series=series, n=horizon_n)

		# format forecast
		forecast_df = forecast.pd_dataframe()
		print(forecast_df.head(10))


	def _get_horizon_n(self):
		""" how many individual predictions in a forecast horizon """
		return NR_PREDS_A_DAY[self.theme] * HORIZON_DAYS[self.forecast_horizon]


