from ecoforecasting.tuning import hyperparam_tuner, named_model
from ecoforecasting.params_suggestions import TranformerSugg

from pytorch_lightning.callbacks import Callback, EarlyStopping

from darts.models import TransformerModel

# fixed hyperparams
STATIC_PARAMS = {
	"batch_size": 1024,
	"max_n_epochs": 70,
	"nr_epochs_val_period": 1,
	"max_samples_per_ts": 1000,
	"input_chunk_length": 356,
}

# initializing objects

# transformer = named_model(model=TransformerModel, model_name="TransformerModel")
# print(transformer)
transformer_tuner = hyperparam_tuner(
	model=TransformerModel,
	model_name="TransformerModel",
	suggest_params_fn=TranformerSugg,
	static_hyperparams=STATIC_PARAMS,
)
