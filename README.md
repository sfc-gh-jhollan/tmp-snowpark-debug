## Forecasting using Facebook's Prophet model in Snowflake Model Registry

This repository is sample example of how to deploy Facebook prophet model using snowflake model registry.

## About Data
This dataset is hourly energy consumption data. PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system serving all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.
The hourly power consumption data comes from PJM's website and are in megawatts (MW).

> [!NOTE]
> Note: For this repository we are only using `PJME_hourly.csv`


## Setup
- Set following environment variables

```
export SF_ACCOUNT=YOUR_ACCOUNT;
export SF_USER=YOUR_SF_USERNAME;
export SF_PASSWORD=<PASS>;
export SF_WAREHOUSE=<WAREHOUSE>'
export SF_ROLE=<ROLE>;
export SF_DATABASE=ML_EXAMPLE_PROJECT;
export SF_SCHEMA=MODELS;
```

- Install ml_project module using `pip install -e .`

- Ensure that proper environment variables are set. Connection setting is loaded from `ml_project/config.py`

- Run `load_data.py` to create necessary databases, schemas, stages and load the csv data into snowflake
- From snowsight UI ensure that all resources are deployed and available

## Running from local environment

- You can run the stored procedure directly using `python3 ml_project/train.py` this will train the model and print the output like below

```
Prophet Model trained successfully. 
Current version is v1.
Model Log Time: 43.44016695022583 secs.
Training time 39.89064002037048 secs.
```

## Running stored proecudure in snowflake environment

- Deploy the stored procedure using `python3 deploy_stored_procedure.py`
- From worksheet run following command
  
```
CALL JEFFHOLLAN_DEMO.COMMON.TRAIN_PROPHET_MODEL()
```

This should give you a output like

```
Prophet Model trained successfully. 
Current version is v2.
Model Log Time: 12.176515340805054 secs.
Training time 108.68230938911438 secs.
```

> [!CAUTION]
> Currently the time taken to train model locally is 39.89 secs and in snowpark same model takes about 108.68 secs