from dataclasses import dataclass

from darts.models.forecasting.forecasting_model import ForecastingModel 
# the base forecasting model class 
# used to specify type, but only decendents have functionality

@dataclass
class named_model:
	model: ForecastingModel
	model_name: str