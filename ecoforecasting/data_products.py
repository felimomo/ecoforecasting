from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import pandas as pd

from ecoforecasting.util import is_theme_assert
from ecoforecasting.data_processing import quick_neon_series

# using 
# https://github.com/eco4cast/Forecast_submissions/blob/main/Generate_forecasts/ARIMA/forecast_model.R
# as 'spiritual guide'

THEME_2_VAR = {
	"aquatics": ["temperature", "oxygen", "chla"],
	"terrestrial_daily": ["nee", "le"],
	"terrestrial_30min": ["nee", "le"],
	"ticks": ["amblyomma_americanum"],
	"phenology": ["gcc_90","rcc_90"],
	"beetles": ["abundance", "richness"]
}

THEME_2_FREQ = { # 'offset aliases'
	"aquatics": "D",
	"terrestrial_daily": "D",
	"terrestrial_30min": "min", # there's no 30min offset alias.
	"ticks": "W",
	"phenology": "D",
	"beetles": "W",
}

@dataclass
class dataprod_metadata:
	theme_name: str
	site_id: str # not checked right now!
	time_col = "datetime" # I guess that'll always be the case

	is_theme_assert(theme_name)

	def variables(self):
		return THEME_2_VAR[self.theme_name]

	def frequency(self):
		if self.theme_name in THEME_2_FREQ.keys():
			return THEME_2_FREQ[self.theme_name]
		else:
			raise ValueError(
				f"forecast_metadata.variables(): '{forecast_metadata.theme_name}' not currently supported for dataprod_metadata.frequency()."
			)
			return []

	def data_url(self):
		return f"https://data.ecoforecast.org/neon4cast-targets/{self.theme_name}/{self.theme_name}-targets.csv.gz"


class dataprod:
	""" neon data product """

	def __init__(
		self, 
		metadata: dataprod_metadata, 
		start_date: pd.TimeStamp = pd.TimeStamp("2020-09-25"),
	):
		self.metadata = metadata
		self.start_date = start_date
		self.data = quick_neon_series(
			site_id = metadata.site_id,
			link = metadata.data_url(),
			freq = metadata.frequency(),
			time_col = metadata.time_col,
			day_avg = False,
			start_date = start_date,
		)

	def use_noaa_cov(self):
		if self.metadata.frequency() not in ["D"]:
			raise ValueError(f"NOAA covariates not implemented for '{self.metadata.frequency()}' frequency data")
		
		self.noaa_covariates = get_noaa(
			site_id = self.site_id,
			day_avg = True,
			freq = self.metadata.frequency()
		)


@dataclass
class forecast_metadata:
	dataprod_meta: dataprod_metadata
	model_id: str
	reference_datetime: datetime
	#
	year = datetime.year
	month = datetime.month
	day = datetime.day

# @dataclass
# class forecast_data:
# 	""" a forecasted series together with metadata """
