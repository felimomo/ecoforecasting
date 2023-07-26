from dataclasses import dataclass
from enum import Enum
from typing import List
from datetime import datetime

# using 
# https://github.com/eco4cast/Forecast_submissions/blob/main/Generate_forecasts/ARIMA/forecast_model.R
# as 'spiritual guide'

class themes(str, Enum):
	"""
	list of possible theme id strings. inheritance from 'str' allows one to compare
	values as strings.
	"""
	aquatics = "aquatics"
	terrestrial_daily = "terrestrial_daily"
	terrestrial_30min = "terrestrial_30min"
	ticks = "ticks"
	phenology = "phenology"
	beetles = "beetles"

@dataclass
class forecast_metadata:
	theme_name: themes
	model_id: str
	site_id: str # not checked right now!
	reference_datetime: datetime
	year = datetime.year
	month = datetime.month
	day = datetime.day

	assert theme_name in set(themes), f"'theme_name' must be in {[v.value for v in themes]}."

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

# @dataclass
# class forecast_data:
# 	""" a forecasted series together with metadata """
