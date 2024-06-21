import json
import os
import time

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from ml_project.config import SNOWFLAKE_CONN, prophet_config
from ml_project.snowparkprophet import SnowParkProphet
from ml_project.utils import get_next_version, log_snowflake_connection_info


def get_model_metrics(y_pred, y_true):
    """Calculates mean squared error,
    mean absolute error and mean absolute percentage error

    Args:
        y_pred (pd.Series): predicted values column from pandas dataframe
        y_true (pd.Series): actual values column from pandas dataframe

    Returns:
        dict: combined dictionary of mean squared error,
    mean absolute error and mean absolute percentage error
    """
    mse = mean_squared_error(y_pred, y_true)
    mae = mean_absolute_error(y_true, y_pred)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"mse": mse, "mae": mae, "mape": mape}


def parse_value(value, param_type):
    """
    Parses the provided value into the specified type.

    Args:
        value (str): The value to be parsed, as a string.
        param_type (type): The target type to which the value should be converted.
                           This can be one of the following:
                           - str: The value remains a string, with special handling
                             for 'auto', 'true', and 'false'.
                           - int: The value is converted to an integer.
                           - float: The value is converted to a float.
                           - list: The value is converted from a JSON string to a list.

    Returns:
        The parsed value in the specified type. If the value is 'null' (case-insensitive),
        None is returned. If the conversion fails for any reason, the original string
        value is returned.

    Raises:
        ValueError: If the value cannot be parsed as an integer when expected.
    """
    if value.lower() == "null":
        return None
    if param_type == str:
        if value.lower() == "auto":
            return "auto"
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            return value
    if param_type == int:
        return int(value)
    if param_type == float:
        return float(value)
    if param_type == list:
        return json.loads(value)
    return value


def get_runtime_params(env: str):
    """
    Checks if any argument was set through Github environment variables
    if the variable is present then it passes to Prophet model. If the variable
    is not defined Prophet model will keep the default value. If the variable is passed
    which is not accepeted by Prophet it is stored in custom_params dictionary

    Returns:
    prophet_params: dictionary of Prophet params passed through Github actions
    custom_params: dictionary of custom params passed through Github Actions
    """
    # Define Prophet parameters and their expected types
    prophet_params_datatype = {
        "growth": str,
        "changepoints": list,
        "n_changepoints": int,
        "changepoint_range": float,
        "yearly_seasonality": str,  # 'auto', True, False, or a number
        "weekly_seasonality": str,  # 'auto', True, False, or a number
        "daily_seasonality": str,  # 'auto', True, False, or a number
        "seasonality_mode": str,
        "seasonality_prior_scale": float,
        "holidays_prior_scale": float,
        "changepoint_prior_scale": float,
        "mcmc_samples": int,
        "interval_width": float,
        "uncertainty_samples": int,
        "stan_backend": str,
    }
    prophet_params = {}
    custom_params = {}
    for key, value in prophet_config[env].items():
        param = key.lower()
        if param in prophet_params_datatype:
            param_type = prophet_params_datatype[param]
            prophet_params[param] = parse_value(value, param_type)
        else:
            custom_params[param] = value
    return prophet_params, custom_params


def train_prophet_model(
    df: pd.DataFrame,
    prophet_params: dict,
    holiday_country_code: str,
    built_in_holidays: bool = False,
) -> Prophet:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        prophet_params (dict): _description_
        holiday_country_code (str): _description_
        built_in_holidays (bool, optional): _description_. Defaults to False.

    Returns:
        Prophet: _description_
    """
    model = Prophet(**prophet_params)

    if built_in_holidays:
        model.add_country_holidays(country_name=holiday_country_code)

    model.fit(df)
    return model


def save_model_to_file(model: Prophet, file: str = "serialized_model_V1.json"):
    """_summary_

    Args:
        model (Prophet): Prophet model to save to file
        file (str, optional): File name of the model. Accepted file pattern is serialized_model*.json
        so that this file is ignored in .gitignore. Defaults to 'serialized_model.json'.
    """
    filename = f"/tmp/prophet_example/{str(os.getpid())}-{file}"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(
        filename,
        "w",
    ) as fout:
        fout.write(model_to_json(model))


def main(session: Session) -> str:
    # Check if environment variable is set if not use prod values
    env = os.getenv("ENVIRONMENT", "prod")

    # Get runtime arguments
    prophet_params, custom_params = get_runtime_params(env=env)
    split_date = custom_params["split_date"]
    training_table = custom_params["training_table_name"]

    # Get the table
    df = session.table(training_table).sort("ts")

    # Split the table for training
    df_filtered = df.filter(df.col("ts") < split_date)

    # Rename columns as required by Prophet
    df_train = df_filtered.to_pandas()
    df_train.rename(columns={"TS": "ds", "PJME_MW": "y"}, inplace=True)
    df_train.set_index("ds", inplace=True, drop=False)

    # Train the model
    start_time = time.time()
    model = train_prophet_model(
        df=df_train,
        prophet_params=prophet_params,
        built_in_holidays=True,
        holiday_country_code=custom_params["holiday_country_code"],
    )
    model_training_time = time.time() - start_time

    # Test the model
    df_test = df.filter(df.col("ts") >= split_date).to_pandas()
    df_test = df_test.rename(columns={"TS": "ds"})
    forecast_df = model.predict(df_test)

    # Get next model version
    snowml_registry = Registry(session)
    next_model_version = get_next_version(snowml_registry, "prophet_example")

    # Save model to file
    save_model_to_file(model=model, file=f"serialized_model_{next_model_version}.json")

    # Change datatype of forecast_df['ds'] to string datatype. As Snowflake ml currently do not
    # support datatime object as input for models
    df_test["ds"] = df_test["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")
    forecast_df["ds"] = forecast_df["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Create model context
    prophet_mc = custom_model.ModelContext(
        artifacts={
            # model file from snowflake stage
            "model_file": f"/tmp/prophet_example/{str(os.getpid())}-serialized_model_{next_model_version}.json"
        }
    )

    # Instantiate the model
    sp_prophet_model = SnowParkProphet(prophet_mc, session)

    # Creates signature schema for model. Basically it maps what input or model takes and what output is given by model
    predict_signature = model_signature.infer_signature(
        input_data=df_test[["ds"]],
        output_data=forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
    )

    # Save model to registry
    snowml_registry = Registry(session)
    start_time = time.time()
    custom_mv = snowml_registry.log_model(
        model=sp_prophet_model,
        model_name="prophet_example",
        version_name=next_model_version,
        conda_dependencies=["prophet"],
        options={"relax_version": False, "embed_local_ml_library": True},
        signatures={"predict": predict_signature},
        comment="Prophet timeseries forecast using the CustomModel API",
    )
    model_metrics = get_model_metrics(forecast_df["yhat"], df_test["PJME_MW"])

    custom_mv.set_metric(metric_name="mse", value=model_metrics["mse"])
    custom_mv.set_metric(metric_name="mae", value=model_metrics["mae"])
    custom_mv.set_metric(metric_name="mape", value=model_metrics["mape"])

    snowml_registry.get_model("prophet_example").default = next_model_version
    model_log_time = time.time() - start_time
    return f"Prophet Model trained successfully. \nCurrent version is {next_model_version}.\nModel Log Time: {model_log_time} secs.\nTraining time {model_training_time} secs."


# This method will be used for local development environment of stored procedure
if __name__ == "__main__":
    with Session.builder.configs(SNOWFLAKE_CONN).getOrCreate() as session:
        # Print the current snowflake connection information
        log_snowflake_connection_info(session=session)
        output = main(session=session)
        print(output)
