# Databricks notebook source
# MAGIC %pip install dowhy networkx --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

catalog = "ryuta"
db = "causal"
model = "root_cause_analysis_scm"

# Make sure that the catalog exists
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# Make sure that the schema exists
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Causal Attributions and Root-Cause Analysis in an Online Shop

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook is an extended and updated version of the corresponding blog post: [Root Cause Analysis with DoWhy, an Open Source Python Library for Causal Machine Learning](https://aws.amazon.com/blogs/opensource/root-cause-analysis-with-dowhy-an-open-source-python-library-for-causal-machine-learning/)
# MAGIC
# MAGIC In this example, we look at an online store and analyze how different factors influence our profit. In particular, we want to analyze an unexpected drop in profit and identify the potential root cause of it. For this, we can make use of Graphical Causal Models (GCM).

# COMMAND ----------

# MAGIC %md
# MAGIC ## The scenario

# COMMAND ----------

# MAGIC %md
# MAGIC Suppose we are selling a smartphone in an online shop with a retail price of $999. The overall profit from the product depends on several factors, such as the number of sold units, operational costs or ad spending. On the other hand, the number of sold units, for instance, depends on the number of visitors on the product page, the price itself and potential ongoing promotions. Suppose we observe a steady profit of our product over the year 2021, but suddenly, there is a significant drop in profit at the beginning of 2022. Why?
# MAGIC
# MAGIC In the following scenario, we will use DoWhy to get a better understanding of the causal impacts of factors influencing the profit and to identify the causes for the profit drop. To analyze our problem at hand, we first need to define our belief about the causal relationships. For this, we collect daily records of the different factors affecting profit. These factors are:
# MAGIC
# MAGIC - **Shopping Event?**: A binary value indicating whether a special shopping event took place, such as Black Friday or Cyber Monday sales.
# MAGIC - **Ad Spend**: Spending on ad campaigns.
# MAGIC - **Page Views**: Number of visits on the product detail page.
# MAGIC - **Unit Price**: Price of the device, which could vary due to temporary discounts.
# MAGIC - **Sold Units**: Number of sold phones.
# MAGIC - **Revenue**: Daily revenue.
# MAGIC - **Operational Cost**: Daily operational expenses which includes production costs, spending on ads, administrative expenses, etc.
# MAGIC - **Profit**: Daily profit.
# MAGIC
# MAGIC Looking at these attributes, we can use our domain knowledge to describe the cause-effect relationships in the form of a directed acyclic graph, which represents our causal graph in the following. The graph is shown here:

# COMMAND ----------

from IPython.display import Image
Image('images/online-shop-graph.png')

# COMMAND ----------

# MAGIC %md
# MAGIC In this scenario we know the following:
# MAGIC
# MAGIC **Shopping Event?** impacts:  
# MAGIC → Ad Spend: To promote the product on special shopping events, we require additional ad spending.  
# MAGIC → Page Views: Shopping events typically attract a large number of visitors to an online retailer due to discounts and various offers.  
# MAGIC → Unit Price: Typically, retailers offer some discount on the usual retail price on days with a shopping event.  
# MAGIC → Sold Units: Shopping events often take place during annual celebrations like Christmas, Father’s day, etc, when people often buy more than usual.  
# MAGIC
# MAGIC **Ad Spend** impacts:  
# MAGIC → Page Views: The more we spend on ads, the more likely people will visit the product page.  
# MAGIC → Operational Cost: Ad spending is part of the operational cost.  
# MAGIC
# MAGIC **Page Views** impacts:  
# MAGIC → Sold Units: The more people visiting the product page, the more likely the product is bought. This is quite obvious seeing that if no one would visit the page, there wouldn’t be any sale.  
# MAGIC
# MAGIC **Unit Price** impacts:  
# MAGIC → Sold Units: The higher/lower the price, the less/more units are sold.  
# MAGIC → Revenue: The daily revenue typically consist of the product of the number of sold units and unit price.  
# MAGIC
# MAGIC **Sold Units** impacts:  
# MAGIC → Sold Units: Same argument as before, the number of sold units heavily influences the revenue.  
# MAGIC → Operational Cost: There is a manufacturing cost for each unit we produce and sell. The more units we well the higher the revenue, but also the higher the manufacturing costs.  
# MAGIC
# MAGIC **Operational Cost** impacts:  
# MAGIC → Profit: The profit is based on the generated revenue minus the operational cost.  
# MAGIC
# MAGIC **Revenue** impacts:  
# MAGIC → Profit: Same reason as for the operational cost.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define causal model

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let us model these causal relationships. In the first step, we need to define a so-called structural causal model (SCM), which is a combination of the causal graph and the underlying generative models describing the data generation process.
# MAGIC
# MAGIC The causal graph can be defined via:

# COMMAND ----------

import networkx as nx

causal_graph = nx.DiGraph([('Page_Views', 'Sold_Units'),
                           ('Revenue', 'Profit'),
                           ('Unit_Price', 'Sold_Units'),
                           ('Unit_Price', 'Revenue'),
                           ('Shopping_Event', 'Page_Views'),
                           ('Shopping_Event', 'Sold_Units'),
                           ('Shopping_Event', 'Unit_Price'),
                           ('Shopping_Event', 'Ad_Spend'),
                           ('Ad_Spend', 'Page_Views'),
                           ('Ad_Spend', 'Operational_Cost'),
                           ('Sold_Units', 'Revenue'),
                           ('Sold_Units', 'Operational_Cost'),
                           ('Operational_Cost', 'Profit')])

# COMMAND ----------

# MAGIC %md
# MAGIC To verify that we did not forget an edge, we can plot this graph:

# COMMAND ----------

import dowhy
from dowhy.utils import plot
plot(causal_graph)

# COMMAND ----------

import json
data = nx.node_link_data(causal_graph)
causal_graph_json = json.dumps(data)

# Writing to causal_graph as json
with open("/databricks/driver/causal_graph.json", "w") as outfile:
    outfile.write(causal_graph_json)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we look at the data from 2021:

# COMMAND ----------

# Custom function defined in 99_utils - loads data from csv files and write them to delta
prepare_data(catalog, db)

# COMMAND ----------

table_name = f"{catalog}.{db}.data_2021"
version_query = f"DESCRIBE HISTORY {table_name}"
version = spark.sql(version_query).collect()[0][0]
sdf = spark.read.format("delta").option("versionAsOf", version).table(table_name)
pdf = sdf.toPandas().set_index("Date")
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC As we see, we have one sample for each day in 2021 with all the variables in the causal graph. Note that in the synthetic data we consider here, shopping events were also generated randomly.
# MAGIC
# MAGIC We defined the causal graph, but we still need to assign generative models to the nodes. We can either manually specify those models, and configure them if needed, or automatically infer “appropriate” models using heuristics from data. We will leverage the latter here:

# COMMAND ----------

from dowhy import gcm

# Create the structural causal model object
scm = gcm.StructuralCausalModel(causal_graph)

# Automatically assign generative models to each node based on the given data
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(
  scm, 
  pdf, 
  override_models=True, 
  quality=gcm.auto.AssignmentQuality.GOOD
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Whenever available, we recommend assigning models based on prior knowledge as then models would closely mimic the physics of the domain, and not rely on nuances of the data. However, here we asked DoWhy to do this for us instead.
# MAGIC
# MAGIC After automatically assign the models, we can print a summary to obtain some insights into the selected models:

# COMMAND ----------

print(auto_assignment_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC As we see, while the auto assignment also considered non-linear models, a linear model is sufficient for most relationships, except for Revenue, which is the product of Sold Units and Unit Price.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fit causal models to data

# COMMAND ----------

# MAGIC %md
# MAGIC After assigning a model to each node, we need to learn the parameters of the model:

# COMMAND ----------

gcm.fit(scm, pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC The fit method learns the parameters of the generative models in each node. Before we continue, let's have a quick look into the performance of the causal mechanisms and how well they capture the distribution:

# COMMAND ----------

print(gcm.evaluate_causal_model(
  scm,
  pdf, 
  compare_mechanism_baselines=True, 
  evaluate_invertibility_assumptions=False))

# COMMAND ----------

# MAGIC %md
# MAGIC The fitted causal mechanisms are fairly good representations of the data generation process, with some minor inaccuracies. However, this is to be expected given the small sample size and relatively small signal-to-noise ratio for many nodes. Most importantly, all the baseline mechanisms did not perform better, which is a good indicator that our model selection is appropriate. Based on the evaluation, we also do not reject the given causal graph.

# COMMAND ----------

# MAGIC %md
# MAGIC > The selection of baseline models or the p-value for graph falsification can be configured as well. For more details, take a look at the corresponding evaluate_causal_model documentation.

# COMMAND ----------

import mlflow

class SCM(mlflow.pyfunc.PythonModel):
  def __init__(self, scm, causal_graph, target_node):
    from dowhy import gcm
    import pandas as pd
    self.scm = scm
    self.causal_graph = causal_graph
    self.target_node = target_node

  def load_scm(self):
    return self.scm
  
  def load_causal_graph(self):
    return self.causal_graph
  
  def predict(self, context, input_df):
    return pd.DataFrame(gcm.attribute_anomalies(self.scm, target_node=self.target_node, anomaly_samples=input_df))

# COMMAND ----------

from dowhy import gcm
import mlflow
import mlflow.data
from mlflow.data.spark_dataset import SparkDataset
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
from mlflow.models import infer_signature

# Define input and output schema
input_example = pdf.iloc[[0]]
signature = infer_signature(
    model_input=input_example, 
    model_output=pd.DataFrame(gcm.attribute_anomalies(scm, target_node="Profit", anomaly_samples=input_example)),
    )
registered_model_name = f"{catalog}.{db}.{model}"

with mlflow.start_run(run_name="causal_model") as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=SCM(scm, causal_graph, "Profit"),
        pip_requirements=[
            "dowhy==" + dowhy.__version__, 
            "networkx==" + nx.__version__,
            ],
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )
    mlflow.log_params({
        **{
            "override_models": "True", 
            "quality": "gcm.auto.AssignmentQuality.GOOD",
        }})
    mlflow.log_artifact("/databricks/driver/causal_graph.json", artifact_path="causal_graph")
    mlflow.log_input(mlflow.data.from_spark(df=sdf, table_name=table_name, version=version), context="training")

# COMMAND ----------

from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

# Assign an alias to the latest model version
def get_latest_model_version(mlflow_client, registered_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


model_version = get_latest_model_version(mlflow_client, registered_model_name)
mlflow_client.set_registered_model_alias(registered_model_name, "champion", model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | dowhy | A Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT | https://pypi.org/project/dowhy/
# MAGIC | networkx | A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. | BSD License | https://pypi.org/project/networkx/

# COMMAND ----------


