import setuptools

setuptools.setup(
	name="ecoforecasting",
	version="0.0.1",
	license="MIT",
	description="Provide forecasting tools for NEON data with NOAA covariates",
	author="Felipe Montealegre-Mora",
	author_email="felimomouni@gmail.com",
	url="https://github.com/felimomo/ecoforecasting",
	keywords=[
		"ecological forecasting",
		"time series forecasting",
		"NEON",
		"NOAA",
	],
	packages=setuptools.find_packages(exclude=["docs", "scripts", "tests"]),
	install_requires=[
		"numpy", "pandas", "polars", "darts", "optuna", "pytorch_lightning", "torch",
	],
	extras_require={
		"tests": [
			# Run tests and coverage
			"pytest",
			"pytest-cov",
			"pytest-env",
			"pytest-xdist",
			# Type check
			"pytype",
			# Lint code
			"flake8>=3.8",
			# Sort imports
			"isort>=5.0",
			# Reformat
			"black",
		],
		"docs": [
			"sphinx",
			"sphinx-autobuild",
			"sphinx-rtd-theme",
			# For spelling
			"sphinxcontrib.spelling",
			# Type hints support
			"sphinx-autodoc-typehints",
		],
		"extra": ["twine"],
	},
	# Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
	],
)