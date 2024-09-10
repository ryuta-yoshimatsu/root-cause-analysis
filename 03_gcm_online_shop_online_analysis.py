# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #Create a model serving endpoint with Python
# MAGIC Now we have a fine-tuned model registered in Unity Catalog, our final step is to deploy this model behind a Model Serving endpoint. This notebook covers wrapping the REST API queries for model serving endpoint creation, updating endpoint configuration based on model version, and endpoint deletion with Python for your Python model serving workflows.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md Specify some variables.

# COMMAND ----------

catalog = "ryuta"
db = "causal"
model = "root_cause_analysis_scm"
log_schema = "log" # A schema within the catalog where the inferece log is going to be stored 
model_name = f"{catalog}.{db}.{model}"  # An existing model in model registry, may have multiple versions
model_serving_endpoint_name = f"root-cause-analysis-{model}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get token and model version
# MAGIC  The following section demonstrates how to provide both a token for the API, which can be obtained from the notebook and how to get the latest model version you plan to serve and deploy.

# COMMAND ----------

token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)

# With the token, you can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Next you need an endpoint at which to execute your request which you can get from the notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# Lastly, extract the Databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

champion_version = client.get_model_version_by_alias(model_name, "champion")
model_version = champion_version.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up configurations
# MAGIC Depending on the latency and throughput requirements of your use case, you want to choose the right `workload_type` and `workload_size`. The `auto_capture_config` block specifies where to write the inference logs: i.e. requests and responses from the endpoint with a timestamp. 

# COMMAND ----------

import requests

my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_type": "CPU",
                "workload_size": "Small",
                "scale_to_zero_enabled": "true",
            }
        ],
        "auto_capture_config": {
            "catalog_name": catalog,
            "schema_name": log_schema,
            "table_name_prefix": model_serving_endpoint_name,
        },
    },
}

# Make sure to the schema for the inference table exists
_ = spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {catalog}.{log_schema}"
)

# Make sure to drop the inference table of it exists
_ = spark.sql(
    f"DROP TABLE IF EXISTS {catalog}.{log_schema}.`{model_serving_endpoint_name}_payload`"
)

# COMMAND ----------

# MAGIC %md
# MAGIC The following defines Python functions that:
# MAGIC - create a model serving endpoint
# MAGIC - update a model serving endpoint configuration with the latest model version
# MAGIC - delete a model serving endpoint

# COMMAND ----------


def func_create_endpoint(model_serving_endpoint_name):
    # get endpoint status
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    r = requests.get(url, headers=headers)
    if "RESOURCE_DOES_NOT_EXIST" in r.text:
        print(
            "Creating this new endpoint: ",
            f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations",
        )
        re = requests.post(endpoint_url, headers=headers, json=my_json)
    else:
        new_model_version = (my_json["config"])["served_models"][0]["model_version"]
        print(
            "This endpoint existed previously! We are updating it to a new config with new model version: ",
            new_model_version,
        )
        # update config
        url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
        re = requests.put(url, headers=headers, json=my_json["config"])
        # wait till new config file in place
        import time, json

        # get endpoint status
        url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
        retry = True
        total_wait = 0
        while retry:
            r = requests.get(url, headers=headers)
            assert (
                r.status_code == 200
            ), f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
            endpoint = json.loads(r.text)
            if "pending_config" in endpoint.keys():
                seconds = 10
                print("New config still pending")
                if total_wait < 6000:
                    # if less the 10 mins waiting, keep waiting
                    print(f"Wait for {seconds} seconds")
                    print(f"Total waiting time so far: {total_wait} seconds")
                    time.sleep(10)
                    total_wait += seconds
                else:
                    print(f"Stopping,  waited for {total_wait} seconds")
                    retry = False
            else:
                print("New config in place now!")
                retry = False

    assert (
        re.status_code == 200
    ), f"Expected an HTTP 200 response, received {re.status_code}. {re}"


def func_delete_model_serving_endpoint(model_serving_endpoint_name):
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    response = requests.delete(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    else:
        print(model_serving_endpoint_name, "endpoint is deleted!")
    return response.json()


# COMMAND ----------

func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for the endpoint to be ready
# MAGIC
# MAGIC The `wait_for_endpoint()` function defined in the following command gets and returns the serving endpoint status.

# COMMAND ----------

import time, mlflow

def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    while True:
        url = f"{endpoint_url}/{model_serving_endpoint_name}"
        response = requests.get(url, headers=headers)
        assert (
            response.status_code == 200
        ), f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("state", {}).get("ready", {})
        # print("status",status)
        if status == "READY":
            print(status)
            print("-" * 80)
            return
        else:
            print(f"Endpoint not ready ({status}), waiting 300 seconds")
            time.sleep(300)  # Wait 300 seconds

api_url = mlflow.utils.databricks_utils.get_webapp_url()

wait_for_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the model
# MAGIC The following cell defines the `generate_image()` function and sends an example generation request under the `payload_json` variable.

# COMMAND ----------

import os
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)

java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
instance = tags["browserHostName"]

# Replace URL with the end point invocation url you get from Model Seriving page.
endpoint_url = (
    f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
)
token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

def generate_attribution(dataset, url=endpoint_url, databricks_token=token):
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()

# COMMAND ----------

anomaly = spark.read.table(f"{catalog}.{db}.data_first_day_2022").toPandas().set_index("Date")
result = generate_attribution(anomaly)
print(result["predictions"][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete the endpoint

# COMMAND ----------

func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | dowhy | A Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT | https://pypi.org/project/dowhy/
# MAGIC | networkx | A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. | BSD License | https://pypi.org/project/networkx/

# COMMAND ----------


