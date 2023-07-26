from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ecoforecasting.util import is_theme_assert

# using 
# https://github.com/eco4cast/Forecast_submissions/blob/main/Generate_forecasts/ARIMA/forecast_model.R
# as 'spiritual guide'

@dataclass
class dataprod_metadata:
	theme_name: str
	site_id: str # not checked right now!

	is_theme_assert(theme_name)

	def variables(self):
		if self.theme_name == "aquatics":
			return ["temperature", "oxygen", "chla"]
		elif (
			(self.theme_name == "terrestrial_daily") | 
			(self.theme_name == "terrestrial_30min")
		):
			return ["nee", "le"]
		elif self.theme_name == "ticks":
			return ["amblyomma_americanum"]
		elif self.theme_name == "phenology":
			return ["gcc_90","rcc_90"]
		elif self.theme_name == "beetles":
			return ["abundance", "richness"]
		else:
			raise ValueError(
				"forecast_metadata.variables(): unrecognized 'forecast_metadata.theme_name' value."
			)
			return []

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
