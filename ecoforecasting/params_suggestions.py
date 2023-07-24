# list of params suggestion functions to be used by hyperparam_tuner objects.
# used by optuna optimizer to optimize hyperparameter choices

def TranformerSugg(trial):
	""" darts.models.TransformerModel """
	return {
		"output_chunk_length": trial.suggest_int("output_chunk_length", 15, 35),
		"d_model": trial.suggest_int("d_model", 56, 72),
		"nhead": trial.suggest_categorical("nhead", [2, 4]),
		"num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 4),
		"num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 4),
		"dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
		"dropout": trial.suggest_float("dropout", 0.2, 0.4),
		"lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
	}

def RFSugg(trial):
	""" darts.models.RandomForest """
	return {
		"output_chunk_length": trial.suggest_int("output_chunk_length", 30, 40),
		"lags": trial.suggest_int("lags", 50, 100),
		"lags_past_covariates": trial.suggest_int("lags_past_covariates", 50, 100),
		"n_estimators": trial.suggest_int("n_estimators", 91, 121),
		"max_depth": trial.suggest_int("max_depth", 3, 10),
	}

def TFTSugg(trial):
	""" darts.models.TFTModel """
	return {
		"output_chunk_length": trial.suggest_int("output_chunk_length", 15, 35),
		"hidden_size": trial.suggest_int("hidden_size", 16, 24),
		"lstm_layers": trial.suggest_int("lstm_layers", 2, 7),
		"num_attention_heads": trial.suggest_int("num_attention_heads", 3, 5),
		"hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 6, 10),
		"dropout": trial.suggest_float("dropout", 0.2, 0.4),
		"lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
	}