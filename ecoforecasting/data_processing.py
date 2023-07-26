import numpy as np
import pandas as pd
import polars as pl

import pyarrow.dataset as ds
from pyarrow import fs

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values

AQUATICS = "https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-expanded-observations.csv.gz"
TERRESTRIAL = "https://data.ecoforecast.org/neon4cast-targets/terrestrial_30min/terrestrial_30min-targets.csv.gz"
TICK = "https://data.ecoforecast.org/neon4cast-targets/ticks/ticks-targets.csv.gz"
PHENOLOGY = "https://data.ecoforecast.org/neon4cast-targets/phenology/phenology-targets.csv.gz"
BEETLE = "https://data.ecoforecast.org/neon4cast-targets/beetles/beetles-targets.csv.gz"

def NOAA_stage3_scan(
    site_id:str = "TREE", 
    variable:str = "TMP"
):
    s3 = fs.S3FileSystem(endpoint_override = "data.ecoforecast.org", anonymous = True)
    path = "neon4cast-drivers/noaa/gefs-v12/stage3/parquet"
    dataset = ds.dataset(path, filesystem=s3)
    return (
        pl.scan_pyarrow_dataset(dataset)
        .filter(pl.col("site_id") == site_id)
        .filter(
          (pl.col("variable") == variable)
        )
        .collect()
        # .pivot(index = "datetime", columns = "variable", values = "prediction")
    )

def day_mean(df, var_to_avg = "TMP", time_col = "datetime", avg_name = "TMP_day_avg"):
    """
    averages values of the column given over the course of a day
    """
    return (
        df
        .with_columns(pl.col(time_col).cast(pl.Date).alias("date"))
        .groupby("date")
        .agg(
            [
                pl.col(var_to_avg).mean().alias(avg_name)
            ]
        )
        .sort("date")
    )

def day_mean_several(df, var_names, time_col = "datetime", avg_name_app = "day_avg"):
    """
    averages values of the column given over the course of a day
    """
    return (
        df
        .with_columns(pl.col(time_col).cast(pl.Date).alias("date"))
        .groupby("date")
        .agg(
            [
                pl.col(var).mean().alias("_".join([var,avg_name_app])) for var in var_names
            ]
        )
        .sort("date")
    )

def pl_to_series(df, time_col = "date", freq = "D"):
    """
    input: polars dataframe
    output: corresponding darts.TimeSeries
    
    for now the easiest thing is to use pandas as a middle-man.
    """
    pd_df = df.to_pandas()
    pd_df[time_col] = pd.to_datetime(pd_df[time_col]).dt.tz_localize(None)
    return (
        TimeSeries.from_dataframe(
            pd_df,
            time_col = time_col,
            freq = freq,
        )
    )

def read_neon(
    site_id, 
    time_col = "datetime", 
    link = TERRESTRIAL,
):
    site_data = pd.read_csv(link)
    site_data[time_col] = pd.to_datetime(site_data["datetime"]).dt.tz_localize(None)
    return (
        site_data
        .loc[site_data["site_id"] == site_id]
    )


def inspect_variables(
    link,
    var_col_names = ["variable"],
):
    site_data = pd.read_csv(link)
    print(f"inspect_variables: FYI, the columns are {site_data.columns}")
    return (
        {
            var: f"{site_data[var].unique()}".replace("\n","") for var in var_col_names
        }
    )

def quick_neon_series(
    site_id,
    link = TERRESTRIAL,
    freq = "D",
    time_col = "datetime",
    day_avg: bool = False,
    start_date = pd.Timestamp("2020-09-25"),
):
    # pandas
    data = read_neon(
        site_id = site_id, 
        time_col = time_col, 
        link = link,
    )
    
    # polars
    data_pl = pl.from_pandas(
        data
        .pivot(index = time_col, columns = "variable", values = "observation")
        .reset_index()
    )
    
    # optional average
    if day_avg:
        freq = "D" # override user-set frequency
        data_pl = day_mean_several(
            data_pl, 
            data["variable"].unique(), 
            time_col = time_col, 
            avg_name_app = "day_avg",
        )
        time_col = "date"
    
    # darts
    data_series = fill_missing_values(
        pl_to_series(data_pl, time_col = time_col, freq = freq)
    )
    
    # split
    pre, data_series_out = data_series.split_before(start_date)
    
    return data_series_out





#################

def get_noaa(
    site_id = "KONZ",
    day_avg = True,
    freq = "D"
):
    s3 = fs.S3FileSystem(endpoint_override = "data.ecoforecast.org", anonymous = True)
    path = f"neon4cast-drivers/noaa/gefs-v12/stage3/parquet/"
    dataset = ds.dataset(path, filesystem=s3, partitioning=["site_id"])

    historic_noaa_full = pl.scan_pyarrow_dataset(dataset) 

    # specialize to a given site, go to pivot
    historic_noaa = (
        historic_noaa_full
        .filter(pl.col("site_id") == site_id)
        .collect()
        .pivot(index = "datetime", columns = "variable", values = "prediction")
    )
    # historic_noaa["datetime"] = pd.to_datetime(historic_noaa["datetime"])
    # historic_noaa.with_columns(pl.col("datetime").cast(pl.Datetime).alias("datetime"))

    # thin out
    historic_noaa = historic_noaa[["datetime", "air_temperature", "air_pressure", "precipitation_flux", "relative_humidity"]]

    # optionally day-average:
    if day_avg:
        freq = "D" # override input as safety because only "D" makes sense here
        historic_noaa_daily = day_mean_several(
            historic_noaa, 
            ["air_temperature", "air_pressure", "precipitation_flux", "relative_humidity"]
        )
        historic_noaa_daily.columns = ["date", "tmp_avg", "pressure_avg", "precip_flux_avg", "rel_humidity_avg"]
        return fill_missing_values( 
            pl_to_series(
                historic_noaa_daily, time_col = "date", freq = freq
            )
        )
    
    return fill_missing_values(
        pl_to_series(
            historic_noaa, time_col = "datetime", freq = freq
        )
    )