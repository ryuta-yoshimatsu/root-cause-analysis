# Databricks notebook source
import numpy as np
import pandas as pd

def clean_df(df):
  
  # Create Spark DataFrame
  df = spark.createDataFrame(df)

  # Rename the column to remove invalid characters
  df = df.withColumnRenamed("Shopping Event?", "Shopping_Event")

  # Rename columns to replace spaces with underscores
  for col in df.columns:
    new_col = col.replace(" ", "_")
    df = df.withColumnRenamed(col, new_col)

  return df


def prepare_data(catalog, db):
  
  # Read into Pandas Dataframe
  data_2021 = pd.read_csv('data/2021 Data.csv')
  data_first_day_2022 = pd.read_csv('data/2022 First Day.csv')
  data_first_quarter_2022 = pd.read_csv('data/2022 First Quarter.csv')

  # Create Spark DataFrame
  data_2021 = clean_df(data_2021)
  data_first_day_2022 = clean_df(data_first_day_2022)
  data_first_quarter_2022 = clean_df(data_first_quarter_2022)

  # Write to Delta
  data_2021.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{db}.data_2021")
  data_first_day_2022.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{db}.data_first_day_2022")
  data_first_quarter_2022.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{db}.data_first_quarter_2022")
  

# COMMAND ----------

#from dowhy.datasets import sales_dataset
#data_2021 = sales_dataset(start_date="2021-01-01", end_date="2021-12-31")
#data_2022 = sales_dataset(start_date="2022-01-01", end_date="2022-12-31", change_of_price=0.9)

# COMMAND ----------


