"""
commonly used utilities for snowpark ml
Ref: https://github.com/cromano8/Snowflake_ML_Intro/blob/main/notebooks/common.py
"""

import json

from snowflake.ml.version import VERSION as SNOWPARK_ML_VERSION
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION as SNOWPARK_VERSION


def get_next_version(reg, model_name) -> str:
    """
    Returns the next version of a model based on the existing versions in the registry.

    Args:
        reg: The registry object that provides access to the models.
        model_name: The name of the model.

    Returns:
        str: The next version of the model in the format "V_<version_number>".

    Raises:
        ValueError: If the version list for the model is empty or if the version format is invalid.
    """
    models = reg.show_models()
    model_name = model_name.upper()
    if models.empty:
        return "v1"
    models = models[models["name"] == model_name]
    if len(models.index) == 0:
        return "v1"
    versions = json.loads(models["versions"][0])
    max_version = max(int(v[1:]) for v in versions)
    return f"v{max_version + 1}"


def log_snowflake_connection_info(session: Session):
    """
    Prints information of snowflake connection

    Args:
        session (snowpark.Session): Current Snowflake session
    """
    snowflake_environment = session.sql(
        "SELECT current_user(), current_version()"
    ).collect()
    snowpark_version = SNOWPARK_VERSION
    snowpark_ml_version = SNOWPARK_ML_VERSION

    # Current Environment Details
    print("\nConnection Established with the following parameters:")
    print("User                        : {}".format(snowflake_environment[0][0]))
    print("Role                        : {}".format(session.get_current_role()))
    print("Database                    : {}".format(session.get_current_database()))
    print("Schema                      : {}".format(session.get_current_schema()))
    print("Warehouse                   : {}".format(session.get_current_warehouse()))
    print("Snowflake version           : {}".format(snowflake_environment[0][1]))
    print(
        "Snowpark for Python version : {}.{}.{}".format(
            snowpark_version[0], snowpark_version[1], snowpark_version[2]
        )
    )
    print("Snowpark ML Version         : {}".format(snowpark_ml_version))
    print("Snowpark ML Version         : {}".format(snowpark_ml_version))
