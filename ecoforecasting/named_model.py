from dataclasses import dataclass
from typing import Type

from darts.models.forecasting.forecasting_model import ForecastingModel 
# the base forecasting model class 
# used to specify type, but only decendents have functionality

@dataclass
class named_model_class:
	model_class: Type[ForecastingModel]
	model_name: str

@dataclass
class named_model:
	model_class: Type[ForecastingModel]
	model_name: str
	model_args: dict
	model_args["model_name"] = model_name
	model_instance = model_class(**model_args)
