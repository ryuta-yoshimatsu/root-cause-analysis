# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Causal Relationships
# MAGIC
# MAGIC In this first notebook, we will explore the example use case, generate a synthetic dataset, create a causal graph, and log that graph to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following or similar specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster
# MAGIC - Databricks Runtime 15.4 LTS ML or above
# MAGIC - Single-node cluster: e.g. `m5d.2xlarge` on AWS or `Standard_D8ds_v5` on Azure Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install and import dependencies

# COMMAND ----------

# DBTITLE 1,Install graphviz from nicer visualization
# MAGIC %sh 
# MAGIC sudo apt-get -qq update
# MAGIC sudo apt-get -y -qq install graphviz libgraphviz-dev

# COMMAND ----------

# MAGIC %md
# MAGIC We install the required libraries from the `requirements.txt`.

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

# Make sure the catalog exists
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# Make sure the schema exists
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# Get the current user name
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
experiment_name = f"/Users/{current_user_name}/rca_manufacturing"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case Study
# MAGIC
# MAGIC In this example, we examine a manufacturing company's production line to see how various factors affect the quality of processed products. In particular, we focus on products flagged as defective by the quality control system and aim to uncover the potential root cause. To do this, we use Graphical Causal Models (GCM).
# MAGIC
# MAGIC Suppose we are responsible for operating a production line in an assembly. The overall quality of the product depends on several checks, such as dimensional verification, torque checks, and visual inspection. For instance, the product’s dimensions rely on the positional and alignment precision of the mechanical process, as well as the forces and torques exerted by machines. These factors, in turn, may be influenced by environmental conditions like humidity or a manual operator. Now imagine that product quality remains steady for a long period, but suddenly there is a significant drop. Why?
# MAGIC
# MAGIC In the following scenario, we will use DoWhy to gain deeper insights into how different factors influence product quality and to identify the causes behind the quality drop. To analyze our problem, we first need to define our assumptions about the causal relationships. For this, we collect measurements of the various factors from our assembly line that may influence product quality. These factors include:
# MAGIC
# MAGIC - **Raw Material**: A binary variable indicating the supplier of the raw material.
# MAGIC - **Material**: A binary variable indicating the supplier of the additional material used in the process.
# MAGIC - **Worker**: A binary variable indicating which manual operator was in charge of the process.
# MAGIC - **Machine**: A binary variable indicating the setting of the machine used in the process.
# MAGIC - **Environment**: Three nodes with continuous variables that describe the conditions in the process chamber (temperature, humidity, and pressure).
# MAGIC - **Position & Alignment**: A continuous variable indicating the extent to which the materials and machine deviate from the standard in terms of positioning and alignment.
# MAGIC - **Force & Torque**: A continuous variable indicating the forces and torques exerted by the machine on the materials.
# MAGIC - **Temperature**: A continuous variable indicating the temperature of either the materials at the interface with the machine.
# MAGIC - **Dimensions**: A binary variable indicating the result of the dimensional verification check on the processed material (0: pass, 1: fail).
# MAGIC - **Torque Checks**: A binary variable indicating the result of the torque-resistance check on the processed material (0: pass, 1: fail).
# MAGIC - **Visual Inspection**: A binary variable indicating the result of the visual inspection on the processed material (0: pass, 1: fail).
# MAGIC - **Quality**: A binary variable reflecting the overall check result. If any prior check fails, this fails as well.

# COMMAND ----------

# MAGIC %md
# MAGIC After discussing with the domain experts, we find the following cause-effect relationships between the variables:
# MAGIC
# MAGIC **Raw Material** impacts:  
# MAGIC → Forces & Torque: Raw materials from different suppliers have slightly different material properties, requiring different forces and torques to process.
# MAGIC
# MAGIC **Material** impacts:  
# MAGIC → Forces & Torque: Materials from different suppliers have slightly different material properties, requiring different forces and torques to process.  
# MAGIC
# MAGIC **Worker** impacts:  
# MAGIC → Position & Alignment: Each worker has different skills and experience, which can affect the precision of materials and machine positioning and alignment.
# MAGIC
# MAGIC **Machine** impacts:   
# MAGIC → Position & Alignment: Each machine setting has varying levels of precision, which can affect the positioning and alignment of the materials and the machine.  
# MAGIC → Forces & Torque: Each machine setting has varying levels of strength, which can affect the forces and torques exerted on the materials.
# MAGIC
# MAGIC **Environment: Chamber Temperature, Chamber Humidity and Chamber Pressure** impact:  
# MAGIC → Temperature: A higher chamber temperature may increase the temperature at the material-machine interface. Conversely, higher chamber humidity and pressure may reduce the interface temperature due to condensation. 
# MAGIC
# MAGIC **Position & Alignment** impact:  
# MAGIC → Dimensions: Imprecise positioning and alignment of the materials and machine may lead to the processed product failing to meet its dimensional requirements.  
# MAGIC
# MAGIC **Forces & Torque** impacts:  
# MAGIC → Dimensions: Weaker forces and torques applied by the machine to the materials may lead to the processed product failing to meet its dimensional requirements.  
# MAGIC → Torque Checks: Weaker forces and torques applied by the machine to the materials may cause the processed product to fail its torque resistance requirements.  
# MAGIC
# MAGIC **Temperature** impacts:  
# MAGIC → Torque Checks: Lower temperature at the material-machine interface may result in weaker joints between materials, causing the processed product to fail its torque resistance requirements.  
# MAGIC → Visual Inspection: Higher temperature at the material-machine interface may result in unwanted appearance at the joints between materials (e.g., welding spatters), causing the processed product to fail its visual inspection requirements.  
# MAGIC
# MAGIC **Dimensions** impacts:  
# MAGIC → Quality: If a product fails the dimensional verification checks, it fails the quality check.  
# MAGIC
# MAGIC **Torque Checks** impacts:  
# MAGIC → Quality: If a product fails the torque resistance checks, it fails the quality check.  
# MAGIC
# MAGIC **Visual Inspection** impacts:  
# MAGIC → Quality: If a product fails the visual insprection checks, it fails the quality check.  
# MAGIC
# MAGIC The attributes and the cause-effect relationships between them can be described in the form of a directed acyclic graph, which represents our causal graph in the following.

# COMMAND ----------

from IPython.display import Image, display
display(Image('./images/manufacturing-process-A-simplified.png', width=1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate data
# MAGIC
# MAGIC Because this is a fictitious use case, we generate a synthetic dataset that aligns with our causal graph. Performing causal analysis on a synthetic dataset allows us to validate the approach and better understand the technique. After generating the dataset, we store it in a Delta table for later use. For more details, refer to the `generate_data` function in the `99_utils` notebook.

# COMMAND ----------

X = generate_data(catalog, schema, 1000)

display(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate a causal graph
# MAGIC
# MAGIC From these relationships, we will construct our causal graph using the `DiGraph` class from the `networkx` package. Here the package `graphviz` and `pygraphviz` give us a nicely formatted DAG show below. 

# COMMAND ----------

true_graph = nx.DiGraph()
true_graph.add_nodes_from = X.columns

dependencies = {
        "position_alignment": ['worker', 'machine'],
        "force_torque": ['raw_material', 'machine', 'material'],
        "temperature": ['chamber_temperature', 'chamber_humidity', 'chamber_pressure'],
        "dimensions": ['position_alignment', 'force_torque'],
        "torque_checks": ['force_torque', 'temperature'],
        "visual_inspection": ['temperature'],
        "quality": ['dimensions', 'torque_checks', 'visual_inspection'],
    }

for child in dependencies:
    for parent in dependencies[child]:
        true_graph.add_edge(parent, child)

dowhy.gcm.util.plot(true_graph, figure_size=(20, 20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the causal graph to MLflow
# MAGIC
# MAGIC Because this graph will be used in subsequent notebooks, we will log it as an artifact using `MLflow`.

# COMMAND ----------

with mlflow.start_run(run_name="causal_graph") as run:
    
    # save graph object to file
    pickle.dump(true_graph, open('/databricks/driver/causal_graph.pickle', 'wb'))

    # log the pickle file to mlflow
    mlflow.log_artifact("/databricks/driver/causal_graph.pickle", artifact_path="graph")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap up
# MAGIC
# MAGIC In this notebook, we explored the example use case of manufacturing production line, generated a synthetic dataset, created a causal graph, and logged that graph to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Appendix: Causal Discovery (WIP)

# COMMAND ----------

from causallearn.search.ConstraintBased.PC import pc

# default parameters
data = X.copy().drop('id', axis=1)
cg = pc(np.vstack(data.to_numpy()), node_names=data.columns)

# visualization using pydot
cg.draw_pydot_graph()

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
