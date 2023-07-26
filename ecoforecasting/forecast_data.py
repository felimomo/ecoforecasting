from dataclasses import dataclass
from enum import Enum
from typing import List
from datetime import datetime

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

# @dataclass
# class forecast_data:
# 	""" a forecasted series together with metadata """
