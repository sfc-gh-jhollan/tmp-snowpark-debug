import pandas as pd
from snowflake.snowpark import Session

session = Session.builder.config("connection_name", "pm").getOrCreate()
session.use_database("JEFFHOLLAN_DEMO")
session.sql("CREATE SCHEMA IF NOT EXISTS COMMON").collect()
session.sql("CREATE SCHEMA IF NOT EXISTS DATA").collect()
session.sql("CREATE SCHEMA IF NOT EXISTS MODELS").collect()
session.sql("CREATE STAGE IF NOT EXISTS COMMON.PYTHON_CODE").collect()

session.use_database("JEFFHOLLAN_DEMO")
session.use_schema("DATA")
# Load data from CSV file
csv_file_path = "data/PJME_hourly.csv"  # Update this with your CSV file path
data = pd.read_csv(csv_file_path)
# Rename columns
data = data.rename(columns={"Datetime": "TS"})

data["TS"] = pd.to_datetime(data["TS"])

# Create a Snowpark DataFrame
df = session.create_dataframe(data)

# Save the DataFrame to a Snowflake table
df.write.save_as_table("PJME_hourly", mode="overwrite")

session.close()
