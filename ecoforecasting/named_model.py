from dataclasses import dataclass

@dataclass
class named_model:
	model: ForecastingModel
	model_name: str