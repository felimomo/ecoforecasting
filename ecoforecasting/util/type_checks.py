THEMES = ["aquatics", "terrestrial_daily", "terrestrial_30min", "ticks", "phenology", "beetles"]

def is_theme_assert(theme_name: str):
	assert theme_name in THEMES, f"'theme_name' must be in {THEMES}."
