import os

SNOWFLAKE_CONN = {
    "account": os.getenv("SF_ACCOUNT", "DEFAULT_ACCOUNT"),
    "user": os.getenv("SF_USER", "DEFAULT_USER"),
    "role": os.getenv("SF_ROLE", "DEFAULT_ROLE"),
    "password": os.getenv("PASSWORD", "DEFAULT_PASSWORD"),
    "database": os.getenv(
        "SF_DATABASE", "ML_EXAMPLE_PROJECT"
    ),  # If possible keep this same otherwise refactor the code accordingly
    "schema": os.getenv(
        "SF_SCHEMA", "MODELS"
    ),  # If possible keep this same otherwise refactor the code accordingly
    "warehouse": os.getenv("SF_WAREHOUSE", "MY_WAREHOUSE_XS"),
}

prophet_config = {
    "prod": {
        "split_date": "2016-04-01",
        "holiday_country_code": "US",
        "include_built_in_holidays": True,
        "training_table_name": "DATA.PJME_HOURLY",  # Training table is under DATA schema
    },
    "dev": {
        "split_date": "2016-04-01",
        "holiday_country_code": "US",
        "include_built_in_holidays": True,
        "training_table_name": "DATA.PJME_HOURLY",  # Training table is under DATA schema
    },
}
