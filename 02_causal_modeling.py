# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Fit Causal Models to Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following or similar specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster
# MAGIC - Databricks Runtime 15.4 LTS ML or above
# MAGIC - Single-node cluster: e.g. `m5d.2xlarge` on AWS or `Standard_D8ds_v5` on Azure Databricks

# COMMAND ----------

# DBTITLE 1,Install graphviz from nicer visualization
# MAGIC %sh 
# MAGIC sudo apt-get -qq update
# MAGIC sudo apt-get -y -qq install graphviz libgraphviz-dev

# COMMAND ----------

# MAGIC %md
# MAGIC We install the required packages from the `requirements.txt`.

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC In the next cell, we run the `99_utils` notebook, which defines a few utility functions that we will use along the way.

# COMMAND ----------

# DBTITLE 1,Run utils notebook
# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define variables and set MLflow experiment

# COMMAND ----------

import mlflow
import pickle
import numpy as np
import pandas as pd
import dowhy
import networkx as nx

# COMMAND ----------

catalog = 'causal_solacc'     # Change this to your catalog name
schema = 'rca'                # Change this to your schema name
model = "scm_manufacturing"   # Change this to your model name

# Check if the catalog exists
catalog_exists = spark.sql(f"SHOW CATALOGS LIKE '{catalog}'").count() > 0
assert catalog_exists, f"Catalog {catalog} does not exist. Run the previous notebook: 01_causal_graph."

# Check if the schema exists
schema_exists = spark.sql(f"SHOW SCHEMAS IN {catalog} LIKE '{schema}'").count() > 0
assert schema_exists, f"Schema {schema} does not exist in catalog {catalog}. Run the previous notebook: 01_causal_graph."

# COMMAND ----------

# Get the current user name
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
experiment_name = f"/Users/{current_user_name}/rca_manufacturing"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the causal graph
# MAGIC
# MAGIC Now, let's load the causal graph defined in the previous notebook. We will integrate this graph with generative models that describe the data generation process at each node to construct a structural causal model (SCM).
# MAGIC
# MAGIC The causal graph can be loaded using:

# COMMAND ----------

# Find all the runs from the prior notebook for causal discovery
client = mlflow.MlflowClient()
experiment = mlflow.get_experiment_by_name(experiment_name)
discovery_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id], 
    filter_string="attributes.run_name='causal_graph'",
    order_by=["start_time DESC"],
    max_results=1,
    )

# Make sure there is at least one run available
assert len(discovery_runs) == 1, "Run the previous notebook: 01_causal_graph"

# The only result should be the latest based on our search_runs call
latest_discovery_run = discovery_runs[0]
latest_discovery_run.info.artifact_uri

# Load the graph artifact from the run
local_path = mlflow.artifacts.download_artifacts(latest_discovery_run.info.artifact_uri + "/graph/causal_graph.pickle")

with open(local_path, "rb") as f:
    causal_graph = pickle.load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC To verify, let's plot the graph:

# COMMAND ----------

dowhy.gcm.util.plot(causal_graph, figure_size=(20, 20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the dataset
# MAGIC
# MAGIC Next, we will load the synthetic dataset we generated in the previous notebook.

# COMMAND ----------

# Define the Delta table name
table_name = f"{catalog}.{schema}.data_manufacturing"

# Query to retrieve the version history of the Delta table
version_query = f"DESCRIBE HISTORY {table_name}"

# Get the latest version of the Delta table
version = spark.sql(version_query).collect()[0][0]

# Read the Delta table as of the latest version
sdf = spark.read.format("delta").option("versionAsOf", version).table(table_name)

# Convert the Spark DataFrame to a pandas DataFrame
pdf = sdf.toPandas()

# Display the first few rows of the pandas DataFrame
pdf.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Assign causal mechanisms to the graph
# MAGIC
# MAGIC As we can see, we have one sample for each processed product, including all the variables in the causal graph.
# MAGIC
# MAGIC While we have defined the causal graph, we still need to assign generative models to its nodes. These models can either be manually specified and configured if necessary or automatically inferred from the data using heuristics. Here, we will use the latter approach:

# COMMAND ----------

from dowhy import gcm
np.random.seed(1)

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
# MAGIC Whenever possible, it is best practice to assign models based on prior knowledge, as this ensures they closely reflect the underlying physics of the domain rather than relying on data-specific nuances. However, in this case, we have asked DoWhy to handle this task for us.
# MAGIC
# MAGIC Once the models are automatically assigned, we can print a summary to gain insights into the selected models:

# COMMAND ----------

print(auto_assignment_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC We see that the auto assignment considered a linear model for non-root nodes with continuous data: e.g., `position_alignment`, whereas for non-root nodes with binary data, it considered a non-linear model: e.g., `dimensions`. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit causal models to the data
# MAGIC
# MAGIC After assigning a model to each node, we need to learn the parameters of the model. We fit the causal model to the dataset.

# COMMAND ----------

gcm.fit(scm, pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the fitted causal models
# MAGIC
# MAGIC The fit method learns the parameters of the generative models in each node. Let's have a look into the performance of the causal mechanisms and how well they capture the distribution:

# COMMAND ----------

print(
  gcm.evaluate_causal_model(
  scm,
  pdf, 
  compare_mechanism_baselines=True, 
  evaluate_invertibility_assumptions=True)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC The fitted causal mechanisms are fairly good representations of the data generation process, with some minor inaccuracies. However, this is to be expected given the small sample size and relatively small signal-to-noise ratio for many nodes. Most importantly, all the baseline mechanisms did not perform better, which is a good indicator that our model selection is appropriate. Based on the evaluation, we also do not reject the given causal graph.

# COMMAND ----------

# MAGIC %md
# MAGIC > The selection of baseline models or the p-value for graph falsification can be configured as well. For more details, take a look at the corresponding evaluate_causal_model documentation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the fitted causal models to Unity Catalog using MLflow

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
    model_output=pd.DataFrame(gcm.attribute_anomalies(scm, target_node="quality", anomaly_samples=input_example)),
    )
registered_model_name = f"{catalog}.{schema}.{model}"

with mlflow.start_run(run_name="causal_model") as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=SCM(scm, causal_graph, "quality"),
        pip_requirements=[
            "dowhy==" + dowhy.__version__, 
            "pandas==" + pd.__version__,
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
    mlflow.log_artifact(local_path, artifact_path="causal_graph")
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
# MAGIC ## Wrap up

# COMMAND ----------

# MAGIC %md
# MAGIC © 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Graphviz | An open source graph visualization software | Common Public License Version 1.0 | https://graphviz.org/download/
# MAGIC | pygraphviz | A Python interface to the Graphviz graph layout and visualization package | BSD | https://pypi.org/project/pygraphviz/
# MAGIC | networkx | A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. | BSD | https://pypi.org/project/networkx/
# MAGIC | dowhy | A Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT | https://pypi.org/project/dowhy/
# MAGIC | causal-learn | A python package for causal discovery that implements both classical and state-of-the-art causal discovery algorithms, which is a Python translation and extension of Tetrad. | MIT | https://pypi.org/project/causal-learn/
